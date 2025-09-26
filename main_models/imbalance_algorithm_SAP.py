import pandas as pd
import numpy as np
from pyomo.environ import *
import os

def run_battery_trading(config, progress_callback=None):
    import os
    # Read Excel sheet
    df = config.input_data.copy()
    datetime_col = None
    for col in df.columns:
        if col.strip().lower() == 'datetime':
            datetime_col = col
            break
    if not datetime_col:
        raise ValueError("No column 'Datetime' or 'datetime' found in the 'Export naar Python' sheet.")
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df.set_index(datetime_col, inplace=True)

    # Use config
    power_mw = config.POWER_MW
    capacity_mwh = config.CAPACITY_MWH
    eff_ch = config.EFF_CH
    eff_dis = config.EFF_DIS
    max_cycles = config.MAX_CYCLES
    time_step_h = config.TIME_STEP_H
    min_soc_frac = config.MIN_SOC
    max_soc_frac = config.MAX_SOC
    min_soc = min_soc_frac * capacity_mwh
    max_soc = max_soc_frac * capacity_mwh
    usable_capacity = capacity_mwh * (max_soc_frac - min_soc_frac)
    
    # Tolerantie voor kleine SoC overschrijdingen (in MWh)
    soc_tolerance = 0.002  # 2 kWh tolerantie
    
    # Haal leveringskosten uit config
    if hasattr(config, 'SUPPLY_COSTS'):
        supply_costs = config.SUPPLY_COSTS
    else:
        supply_costs = 20.0  # Default waarde voor backward compatibility
    
    # Haal transportkosten uit config
    if hasattr(config, 'TRANSPORT_COSTS'):
        transport_costs = config.TRANSPORT_COSTS
    else:
        transport_costs = 15.0  # Default waarde voor backward compatibility

    # Initialization
    total_days = len(df.groupby(pd.Grouper(freq='D')))
    base_daily_cycle_budget = max_cycles / total_days
    remaining_cycles = max_cycles
    vol_window = []

    results = []
    # Make initial SoC configurable
    if hasattr(config, 'INIT_SOC'):
        current_soc = float(config.INIT_SOC) * capacity_mwh
    else:
        current_soc = 0.5 * capacity_mwh
    cumulative_cycles = 0
    cycle_history = []
    
    # Lijst om infeasible dagen bij te houden
    infeasible_days = []

    for day, day_data in df.groupby(pd.Grouper(freq='D')):
        if len(day_data) == 0:
            continue

        msg = f"Optimizing {day.strftime('%d-%m-%Y')}... Cycles used: {cumulative_cycles:.1f}/{max_cycles}"
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg, end='\r')

        # Create model
        model = ConcreteModel()
        T = len(day_data)
        model.T = RangeSet(0, T-1)

        # Binary variable: 1 = charging, 0 = discharging
        model.charge_state = Var(model.T, within=Binary)

        # Determine maximum charge and discharge power per timestep (based on the available space on the grid connection)
        def charge_bound(model, t):
            space_mw = (day_data["space available for charging (kWh)"].iloc[t] / 0.25) / 1000
            verplicht_laden = 0
            space_dis = day_data["space available for discharging (kWh)"].iloc[t]
            if space_dis < 0:
                verplicht_laden = abs(space_dis) / 0.25 / 1000
            # Upper bound is at least the required value (if applicable)
            upper = max(min(power_mw, space_mw), verplicht_laden)
            return (0, upper)
        def discharge_bound(model, t):
            space_mw = (day_data["space available for discharging (kWh)"].iloc[t] / 0.25) / 1000
            verplicht_ontladen = 0
            space_ch = day_data["space available for charging (kWh)"].iloc[t]
            if space_ch < 0:
                verplicht_ontladen = abs(space_ch) / 0.25 / 1000
            upper = max(min(power_mw, space_mw), verplicht_ontladen)
            return (0, upper)
        model.charge = Var(model.T, within=NonNegativeReals, bounds=charge_bound)
        model.discharge = Var(model.T, within=NonNegativeReals, bounds=discharge_bound)
        model.soc = Var(model.T, within=NonNegativeReals, bounds=(min_soc, max_soc))
        model.soc[0].fix(current_soc)

        # Constraint: never charge and discharge at the same time
        M = power_mw  # Large enough
        def no_simultaneous_charge(model, t):
            return model.charge[t] <= M * model.charge_state[t]
        model.no_simul_charge = Constraint(model.T, rule=no_simultaneous_charge)
        def no_simultaneous_discharge(model, t):
            return model.discharge[t] <= M * (1 - model.charge_state[t])
        model.no_simul_discharge = Constraint(model.T, rule=no_simultaneous_discharge)

        def soc_balance(model, t):
            if t == 0: return Constraint.Skip
            # Charging: only a part of the energy ends up in the battery (eff_ch)
            # Discharging: you need to take more out of the battery to deliver 1 MWh to the grid (1/eff_dis)
            return model.soc[t] == model.soc[t-1] + (
                model.charge[t-1] * time_step_h * eff_ch -
                model.discharge[t-1] * time_step_h / eff_dis)
        model.soc_con = Constraint(model.T, rule=soc_balance)

        # Hard constraint: mandatory discharge when negative space available for charging
        def enforce_discharge(model, t):
            space_ch = day_data["space available for charging (kWh)"].iloc[t]
            if space_ch < 0:
                verplicht_ontladen = abs(space_ch) / 0.25 / 1000
                return model.discharge[t] >= verplicht_ontladen
            else:
                return Constraint.Skip
        model.enforce_discharge = Constraint(model.T, rule=enforce_discharge)

        # Hard constraint: mandatory charge when negative space available for discharging
        def enforce_charge(model, t):
            space_dis = day_data["space available for discharging (kWh)"].iloc[t]
            if space_dis < 0:
                verplicht_laden = abs(space_dis) / 0.25 / 1000
                return model.charge[t] >= verplicht_laden
            else:
                return Constraint.Skip
        model.enforce_charge = Constraint(model.T, rule=enforce_charge)

        imbalance_prices = day_data[['price_shortage', 'price_surplus']].max(axis=1)
        today_volatility = imbalance_prices.std()
        vol_window.append(today_volatility)
        if len(vol_window) > 7: vol_window.pop(0)
        rolling_mean = np.mean(vol_window) if len(vol_window) >= 3 else today_volatility

        relative_vol = today_volatility / (rolling_mean + 1e-6)
        scaling_factor = min(1.5, max(0.5, relative_vol))
        daily_cycle_budget = min(base_daily_cycle_budget * scaling_factor, remaining_cycles)

        def daily_cycles(model):
            daily_charge = sum(model.charge[t] for t in model.T) * time_step_h * eff_ch
            daily_discharge = sum(model.discharge[t] for t in model.T) * time_step_h * eff_dis
            return (daily_charge + daily_discharge) / (2 * usable_capacity) <= daily_cycle_budget
        model.cycle_con = Constraint(rule=daily_cycles)

        def objective(model):
            return sum(
                day_data["price_surplus"].iloc[t] * model.discharge[t] * time_step_h * eff_dis -
                day_data["price_shortage"].iloc[t] * model.charge[t] * time_step_h * eff_ch
                for t in model.T if day_data["regulation_state"].iloc[t] != 2
            )
        model.obj = Objective(rule=objective, sense=maximize)

        model.final_soc_min = Constraint(expr=model.soc[T-1] + (
            model.charge[T-1] * time_step_h * eff_ch - model.discharge[T-1] * time_step_h * eff_dis) >= min_soc)

        model.final_soc_max = Constraint(expr=model.soc[T-1] + (
            model.charge[T-1] * time_step_h * eff_ch - model.discharge[T-1] * time_step_h * eff_dis) <= max_soc)

        print(f"{day.strftime('%d-%m-%Y')} | Vol={today_volatility:.2f} | Budget={daily_cycle_budget:.2f} | Remaining={remaining_cycles:.1f}")
        if progress_callback:
            progress_callback(f"{day.strftime('%d-%m-%Y')} | Vol={today_volatility:.2f} | Budget={daily_cycle_budget:.2f} | Remaining={remaining_cycles:.1f}")

        # Solve
        # --- Start of new, robust code ---
        
        solver = None # Initialize solver as None
        
        # 1. First, try to use the local Windows executable.
        #    This might work on your local computer.
        local_cbc_path = os.path.join(os.path.dirname(__file__), 'Cbc-releases.2.10.12-w64-msvc16-md', 'bin', 'cbc.exe')
        
        if os.path.exists(local_cbc_path):
            try:
                solver = SolverFactory('cbc', executable=local_cbc_path)
                print("Using local CBC solver.")
            except Exception as e:
                print(f"Failed to use local CBC solver: {e}")
                solver = None # Ensure solver is None if it fails
        
        # 2. If the first attempt failed (solver is still None), fall back to the system solver.
        #    This will work on your web app's Linux server.
        if solver is None:
            try:
                solver = SolverFactory('cbc')
                print("Using system-wide CBC solver.")
            except ApplicationError:
                # This error is raised if no solver can be found at all
                raise ValueError(
                    "CBC solver not found. Ensure it is installed and in your system's PATH, "
                    "or include the executable with your app."
                )
        
        # Now, you can safely use the solver
        result = solver.solve(model)
        
        # --- End of new code ---

        # Check for infeasibility
        if (result.solver.status != SolverStatus.ok) or (result.solver.termination_condition == TerminationCondition.infeasible):
            # Controleer eerst of het een kleine SoC overschrijding betreft die we kunnen tolereren
            soc_violation_detected = False
            
            # Simuleer minimale vereiste acties
            try:
                test_soc = current_soc
                for t in range(T):
                    space_ch = day_data["space available for charging (kWh)"].iloc[t]
                    space_dis = day_data["space available for discharging (kWh)"].iloc[t]
                    
                    if space_ch < 0:  # Verplicht ontladen
                        min_discharge = abs(space_ch) / 0.25 / 1000  # MW
                        test_soc -= min_discharge * time_step_h / eff_dis
                    
                    if space_dis < 0:  # Verplicht laden
                        min_charge = abs(space_dis) / 0.25 / 1000  # MW
                        test_soc += min_charge * time_step_h * eff_ch
                
                # Check of SoC binnen tolerantie blijft
                if (min_soc - soc_tolerance) <= test_soc <= (max_soc + soc_tolerance):
                    if test_soc < min_soc or test_soc > max_soc:
                        soc_violation_detected = True  # Kleine overschrijding
                
            except:
                soc_violation_detected = False
            
            # Als het een tolereerbare SoC overschrijding is, ga door met dummy resultaten
            if soc_violation_detected:
                dummy_charge = [0.0] * T
                dummy_discharge = [0.0] * T
                dummy_soc = [current_soc] * T
                dummy_revenue = [0.0] * T
                dummy_supplier_costs = [0.0] * T
                
                day_data = day_data.copy()
                day_data['space_for_charging_kWh'] = day_data['space available for charging (kWh)']
                day_data['space_for_discharging_kWh'] = day_data['space available for discharging (kWh)']
                day_data['energy_charged_kWh'] = [c * time_step_h * 1000 for c in dummy_charge]
                day_data['energy_discharged_kWh'] = [d * time_step_h * 1000 for d in dummy_discharge]
                day_data['SoC_kWh'] = [s * 1000 for s in dummy_soc]
                day_data['SoC_pct'] = [(s - min_soc) / (max_soc - min_soc) for s in dummy_soc]
                # Voeg grid_exchange_kWh toe voor SoC violation dagen
                if 'load' in day_data.columns and 'production_PV' in day_data.columns:
                    dummy_energy_charged_kwh = [c * time_step_h * 1000 for c in dummy_charge]
                    dummy_energy_discharged_kwh = [d * time_step_h * 1000 for d in dummy_discharge]
                    day_data['grid_exchange_kWh'] = (day_data['load'].values 
                                                   - day_data['production_PV'].values 
                                                   + np.array(dummy_energy_charged_kwh)      # Dummy laden (0)
                                                   - np.array(dummy_energy_discharged_kwh))  # Dummy ontladen (0)
                else:
                    day_data['grid_exchange_kWh'] = 0
                
                # E-programma is altijd 0 voor SAP
                day_data['e_program_kWh'] = 0
                
                # Day-ahead result is altijd 0 voor SAP
                day_data['day_ahead_result'] = 0
                
                # Imbalance result voor SoC violation dagen
                day_data['imbalance_result'] = dummy_revenue
                day_data['supplier_costs'] = np.array(dummy_supplier_costs)
                results.append(day_data)
                
                warning_msg = f"Let op: Dag {day.strftime('%d-%m-%Y')} - kleine SoC overschrijding getolereerd, batterij niet gebruikt"
                if progress_callback:
                    progress_callback(warning_msg)
                
                continue
            # Voeg deze dag toe aan de lijst van probleemdagen
            dag_datum = day.strftime('%d-%m-%Y')
            reden = "Onbekende reden"
            
            # Probeer de reden te achterhalen
            problemen = []
            
            for t in range(T):
                space_ch = day_data["space available for charging (kWh)"].iloc[t]
                space_dis = day_data["space available for discharging (kWh)"].iloc[t]
                tijd = day_data.index[t]
                
                # Check voor negatieve waarden (overschrijdingen gecontracteerd vermogen)
                if space_ch < 0:
                    verplicht_ontladen = abs(space_ch) / 0.25 / 1000
                    if verplicht_ontladen > power_mw:
                        problemen.append(f"Overschrijding gecontracteerd vermogen: vereist ontladen {verplicht_ontladen:.2f} MW > batterijvermogen {power_mw} MW op {tijd.strftime('%H:%M')}")
                    else:
                        problemen.append(f"Overschrijding gecontracteerd vermogen: verplicht ontladen {verplicht_ontladen:.2f} MW op {tijd.strftime('%H:%M')}")
                
                if space_dis < 0:
                    verplicht_laden = abs(space_dis) / 0.25 / 1000
                    if verplicht_laden > power_mw:
                        problemen.append(f"Overschrijding gecontracteerd vermogen: vereist laden {verplicht_laden:.2f} MW > batterijvermogen {power_mw} MW op {tijd.strftime('%H:%M')}")
                    else:
                        problemen.append(f"Overschrijding gecontracteerd vermogen: verplicht laden {verplicht_laden:.2f} MW op {tijd.strftime('%H:%M')}")
                
                # Check voor gelijktijdige verplichtingen
                if space_ch < 0 and space_dis < 0:
                    problemen.append(f"Conflicterende eisen: gelijktijdig verplicht laden EN ontladen op {tijd.strftime('%H:%M')}")
            
            # Check voor SoC-gerelateerde problemen
            if len(problemen) == 0:
                # Mogelijk SoC probleem - check begin en eind SoC
                if current_soc <= min_soc + 0.01:  # Bijna leeg
                    problemen.append(f"Batterij bijna leeg (SoC: {current_soc:.2f} MWh) en kan vereiste ontlading niet leveren")
                elif current_soc >= max_soc - 0.01:  # Bijna vol
                    problemen.append(f"Batterij bijna vol (SoC: {current_soc:.2f} MWh) en kan vereiste lading niet opnemen")
            
            # Bepaal de hoofdreden
            if len(problemen) > 0:
                reden = problemen[0]  # Eerste probleem als hoofdreden
                if len(problemen) > 1:
                    reden += f" (en {len(problemen)-1} andere problemen)"
            else:
                reden = "Infeasible optimalisatieprobleem - mogelijk conflicterende constraints of onhaalbare combinatie van eisen"
            
            infeasible_days.append({
                'datum': dag_datum,
                'reden': reden
            })
            
            # Log de waarschuwing
            warning_msg = f"WAARSCHUWING: Dag {dag_datum} overgeslagen - {reden}"
            if progress_callback:
                progress_callback(warning_msg)
            else:
                print(warning_msg)
            
            # Creëer dummy resultaten voor deze dag (batterij doet niets)
            dummy_charge = [0.0] * T
            dummy_discharge = [0.0] * T
            dummy_soc = [current_soc] * T
            dummy_revenue = [0.0] * T
            
            # Bereken supplier_costs voor dummy dag (batterij doet niets, dus alleen based op normale grid exchange)
            dummy_energy_charged_kwh = [c * time_step_h * 1000 for c in dummy_charge]
            dummy_energy_discharged_kwh = [d * time_step_h * 1000 for d in dummy_discharge]
            dummy_supplier_costs = [(charge_kwh - discharge_kwh) / 1000 * supply_costs for charge_kwh, discharge_kwh in zip(dummy_energy_charged_kwh, dummy_energy_discharged_kwh)]
            
            # Bereken transportkosten voor dummy dag
            if 'load' in day_data.columns and 'production_PV' in day_data.columns:
                dummy_net_grid_consumption_kwh = (day_data['load'].values 
                                                 - day_data['production_PV'].values 
                                                 + np.array(dummy_energy_charged_kwh)      # Dummy laden (0)
                                                 - np.array(dummy_energy_discharged_kwh))  # Dummy ontladen (0)
                dummy_transport_costs = [max(0, net_consumption) / 1000 * transport_costs for net_consumption in dummy_net_grid_consumption_kwh]
            else:
                dummy_transport_costs = [0.0] * len(dummy_energy_charged_kwh)
            
            # Voeg dummy data toe aan resultaten
            day_data = day_data.copy()
            day_data['space_for_charging_kWh'] = day_data['space available for charging (kWh)']
            day_data['space_for_discharging_kWh'] = day_data['space available for discharging (kWh)']
            day_data['energy_charged_kWh'] = dummy_energy_charged_kwh
            day_data['energy_discharged_kWh'] = dummy_energy_discharged_kwh
            day_data['SoC_kWh'] = [s * 1000 for s in dummy_soc]
            day_data['SoC_pct'] = [(s - min_soc) / (max_soc - min_soc) for s in dummy_soc]
            # Voeg grid_exchange_kWh toe voor dummy dagen
            if 'load' in day_data.columns and 'production_PV' in day_data.columns:
                day_data['grid_exchange_kWh'] = (day_data['load'].values 
                                               - day_data['production_PV'].values 
                                               + np.array(dummy_energy_charged_kwh)      # Dummy laden (0)
                                               - np.array(dummy_energy_discharged_kwh))  # Dummy ontladen (0)
            else:
                day_data['grid_exchange_kWh'] = np.array(dummy_energy_charged_kwh) - np.array(dummy_energy_discharged_kwh)
            
            # E-programma is altijd 0 voor SAP
            day_data['e_program_kWh'] = 0
            
            # Day-ahead result is altijd 0 voor SAP
            day_data['day_ahead_result'] = 0
            
            # Imbalance result voor dummy dagen
            day_data['imbalance_result'] = dummy_revenue
            day_data['supplier_costs'] = np.array(dummy_supplier_costs)
            day_data['transport_costs'] = dummy_transport_costs
            results.append(day_data)
            
            # Ga door naar de volgende dag
            continue

        final_charge = model.charge[T-1]()
        final_discharge = model.discharge[T-1]()
        final_soc = model.soc[T-1]() + (
            final_charge * time_step_h * eff_ch - final_discharge * time_step_h * eff_dis)
        final_soc = min(max(final_soc, min_soc), max_soc)
        current_soc = final_soc

        charge = [model.charge[t]() for t in model.T]
        discharge = [model.discharge[t]() for t in model.T]
        soc = [model.soc[t]() for t in model.T]

        # New definition cycles: only charged energy counts
        charged_energy = sum(charge) * time_step_h  # in MWh
        daily_cycle = (charged_energy * eff_ch) / usable_capacity
        cumulative_cycles += daily_cycle
        remaining_cycles -= daily_cycle
        remaining_cycles = max(0, remaining_cycles)
        cycle_history.append(daily_cycle)

        # Trading result is always based on energy to/from the grid
        revenue = (
            day_data["price_surplus"].values * discharge * time_step_h -
            day_data["price_shortage"].values * charge * time_step_h
        )

        # Energy per timestep in kWh/15min (to/from grid)
        energy_charged = [c * time_step_h * 1000 for c in charge]  # kWh from grid to battery
        energy_discharged = [d * time_step_h * 1000 for d in discharge]  # kWh from battery to grid
        # SoC as a fraction of usable_capacity
        soc_frac = [(s - min_soc) / (max_soc - min_soc) for s in soc]

        # Bereken supplier_costs: kosten voor netto energie die naar/van de batterij gaat
        # Positief = laden (kosten), negatief = ontladen (besparingen)
        supplier_costs = [(charge_kwh - discharge_kwh) / 1000 * supply_costs for charge_kwh, discharge_kwh in zip(energy_charged, energy_discharged)]
        
        # Bereken transportkosten: alleen voor netto afname van het net
        # Voor SAP algoritme: bereken totaal netto grid verbruik (load + batterij_laden - batterij_ontladen)
        # Gebruik de load en PV data uit day_data indien beschikbaar
        if 'load' in day_data.columns and 'production_PV' in day_data.columns:
            # Netto grid verbruik = load - PV + batterij_laden - batterij_ontladen
            net_grid_consumption_kwh = (day_data['load'].values 
                                       - day_data['production_PV'].values 
                                       + np.array(energy_charged)      # Laden = extra afname
                                       - np.array(energy_discharged))  # Ontladen = minder afname
            # Transportkosten alleen voor positieve (afname) uitwisseling
            transport_costs_list = [max(0, net_consumption) / 1000 * transport_costs for net_consumption in net_grid_consumption_kwh]
        else:
            # Fallback: geen PV data beschikbaar, geen transportkosten
            transport_costs_list = [0.0] * len(energy_charged)

        # Prepare columns for output - AANGEPAST VOOR JUISTE KOLOMNAMEN
        soc_kwh = [s * 1000 for s in soc]  # SoC in kWh
        day_data = day_data.copy()
        
        # Voeg de juiste kolommen toe met lowercase namen zoals verwacht door run_model.py
        day_data['space_for_charging_kWh'] = day_data['space available for charging (kWh)']
        day_data['space_for_discharging_kWh'] = day_data['space available for discharging (kWh)']
        day_data['energy_charged_kWh'] = energy_charged
        day_data['energy_discharged_kWh'] = energy_discharged
        day_data['SoC_kWh'] = soc_kwh
        day_data['SoC_pct'] = soc_frac
        # Voeg grid_exchange_kWh toe (netto grid uitwisseling)
        if 'load' in day_data.columns and 'production_PV' in day_data.columns:
            day_data['grid_exchange_kWh'] = (day_data['load'].values 
                                           - day_data['production_PV'].values 
                                           + np.array(energy_charged)      # Laden = extra afname
                                           - np.array(energy_discharged))  # Ontladen = minder afname
        else:
            day_data['grid_exchange_kWh'] = np.array(energy_charged) - np.array(energy_discharged)
        
        # E-programma is altijd 0 voor SAP (alleen batterij op SAP markt)
        day_data['e_program_kWh'] = 0
        
        # Day-ahead result is altijd 0 voor SAP (geen day-ahead trading)
        day_data['day_ahead_result'] = 0
        
        # Imbalance result is de revenue uit onbalanshandel
        day_data['imbalance_result'] = revenue
        day_data['energy_tax'] = 0  # Niet van toepassing voor SAP (0 blijft 0)
        day_data['supplier_costs'] = -np.abs(np.array(supplier_costs))  # Altijd negatief (kosten)
        day_data['transport_costs'] = -np.abs(np.array(transport_costs_list))  # Altijd negatief (kosten)
        
        # Total result = onbalans revenue + alle extra kosten
        total_result = (revenue + 
                       day_data['energy_tax'] + 
                       day_data['supplier_costs'] + 
                       day_data['transport_costs'])
        day_data['total_result_imbalance_SAP'] = total_result
        results.append(day_data)

    final_df = pd.concat(results)

    total_revenue = final_df["total_result_imbalance_SAP"].sum()
    total_cycles = cumulative_cycles

    # Voeg infeasible dagen toe aan summary
    summary = {
        "total_revenue": total_revenue,
        "total_cost": 0,
        "total_cycles": total_cycles,
        "cycle_history": cycle_history,
        "battery_power_MW": power_mw,
        "revenue_per_MW": total_revenue / power_mw if power_mw else float('nan'),
        "infeasible_days": infeasible_days
    }

    return final_df, summary




