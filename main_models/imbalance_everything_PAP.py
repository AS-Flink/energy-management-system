import pandas as pd
import numpy as np
from pyomo.environ import *

import os

def get_energy_tax_table():
    """
    Retourneert energiebelasting tabel gebaseerd op jaarverbruik
    Gebaseerd op Nederlandse energiebelasting 2024 (€/MWh)
    """
    return {
        'consumption_brackets': [
            {'min_mwh': 0, 'max_mwh': 10, 'tax_eur_per_mwh': 101.54},      # 0-10 MWh
            {'min_mwh': 10, 'max_mwh': 50, 'tax_eur_per_mwh': 69.37},      # 10-50 MWh  
            {'min_mwh': 50, 'max_mwh': 10000, 'tax_eur_per_mwh': 38.68},   # 50-10.000 MWh
            {'min_mwh': 10000, 'max_mwh': float('inf'), 'tax_eur_per_mwh': 3.21}  # >10.000 MWh
        ]
    }

def calculate_energy_tax(total_consumption_mwh, tax_table):
    """
    Bereken de marginale energiebelasting voor een bepaald verbruiksniveau
    """
    for bracket in tax_table['consumption_brackets']:
        if bracket['min_mwh'] <= total_consumption_mwh < bracket['max_mwh']:
            return bracket['tax_eur_per_mwh']
    # Fallback naar hoogste bracket
    return tax_table['consumption_brackets'][-1]['tax_eur_per_mwh']

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
    
    # Haal transportkosten uit config
    if hasattr(config, 'TRANSPORT_COSTS'):
        transport_costs = config.TRANSPORT_COSTS
    else:
        transport_costs = 15.0  # Default waarde voor backward compatibility
    
    # Haal e-programma uit config (als percentage van werkelijke afname/invoeding)
    if hasattr(config, 'E_PROGRAM'):
        e_program_percentage = config.E_PROGRAM
    else:
        e_program_percentage = 100.0  # Default 100% = geen afwijking
    
    # Haal leveringskosten uit config
    if hasattr(config, 'SUPPLY_COSTS'):
        supply_costs = config.SUPPLY_COSTS
    else:
        supply_costs = 20.0  # Default waarde voor backward compatibility
    
    # Energiebelasting berekening
    tax_table = get_energy_tax_table()
    if len(df) > 0:
        # Bereken geschat jaarverbruik op basis van gemiddelde load
        avg_load_kwh = df['load'].mean() if 'load' in df.columns else 0
        hours_per_year = 8760
        estimated_annual_consumption_mwh = (avg_load_kwh * hours_per_year) / 1000
        
        # Bereken marginale energiebelasting
        marginal_tax_rate = calculate_energy_tax(estimated_annual_consumption_mwh, tax_table)
        
        if progress_callback:
            progress_callback(f"Geschat jaarverbruik: {estimated_annual_consumption_mwh:.1f} MWh")
            progress_callback(f"Energiebelasting: €{marginal_tax_rate:.2f}/MWh")
            progress_callback(f"Leveringskosten: €{supply_costs:.2f}/MWh")
    else:
        marginal_tax_rate = tax_table['consumption_brackets'][0]['tax_eur_per_mwh']

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

        # Zet inputdata om naar numpy arrays voor Pyomo Param
        pv = day_data["production_PV"].values / 1000  # MWh
        load = day_data["load"].values / 1000         # MWh
        price_shortage = day_data["price_shortage"].values
        price_surplus = day_data["price_surplus"].values
        
        # Bereken volatiliteit en dagelijks cycle budget (zoals in SAP)
        imbalance_prices = day_data[['price_shortage', 'price_surplus']].max(axis=1)
        today_volatility = imbalance_prices.std()
        vol_window.append(today_volatility)
        if len(vol_window) > 7: vol_window.pop(0)
        rolling_mean = np.mean(vol_window) if len(vol_window) >= 3 else today_volatility

        relative_vol = today_volatility / (rolling_mean + 1e-6)
        scaling_factor = min(1.5, max(0.5, relative_vol))
        daily_cycle_budget = min(base_daily_cycle_budget * scaling_factor, remaining_cycles)

        print(f"{day.strftime('%d-%m-%Y')} | Vol={today_volatility:.2f} | Budget={daily_cycle_budget:.2f} | Remaining={remaining_cycles:.1f}")
        if progress_callback:
            progress_callback(f"{day.strftime('%d-%m-%Y')} | Vol={today_volatility:.2f} | Budget={daily_cycle_budget:.2f} | Remaining={remaining_cycles:.1f}")
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
        M = power_mw
        def no_simultaneous_charge(model, t):
            return model.charge[t] <= M * model.charge_state[t]
        model.no_simul_charge = Constraint(model.T, rule=no_simultaneous_charge)
        def no_simultaneous_discharge(model, t):
            return model.discharge[t] <= M * (1 - model.charge_state[t])
        model.no_simul_discharge = Constraint(model.T, rule=no_simultaneous_discharge)

        def soc_balance(model, t):
            if t == 0: return Constraint.Skip
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

        # Cycles constraint met daily budget
        def daily_cycles(model):
            daily_charge = sum(model.charge[t] for t in model.T) * time_step_h * eff_ch
            daily_discharge = sum(model.discharge[t] for t in model.T) * time_step_h * eff_dis
            return (daily_charge + daily_discharge) / (2 * usable_capacity) <= daily_cycle_budget
        model.cycle_con = Constraint(rule=daily_cycles)

        # Netto netpositie per timestep als Pyomo variabelen
        model.netpos = Var(model.T, within=Reals)
        model.netpos_pos = Var(model.T, within=NonNegativeReals)
        model.netpos_neg = Var(model.T, within=NonNegativeReals)
        
        # E-programma en onbalans variabelen
        model.imbalance = Var(model.T, within=Reals)
        model.imbalance_pos = Var(model.T, within=NonNegativeReals)  # Meer afgenomen dan voorspeld
        model.imbalance_neg = Var(model.T, within=NonNegativeReals)  # Meer ingevoed dan voorspeld
        
        # Zet data om naar Pyomo Parameters
        model.load_profile = Param(model.T, initialize={t: float(load[t]) for t in range(T)})
        model.pv = Param(model.T, initialize={t: float(pv[t]) for t in range(T)})
        model.price_shortage = Param(model.T, initialize={t: float(price_shortage[t]) for t in range(T)})
        model.price_surplus = Param(model.T, initialize={t: float(price_surplus[t]) for t in range(T)})
        
        # Day-ahead prijzen
        price_day_ahead = day_data["price_day_ahead"].values
        model.price_day_ahead = Param(model.T, initialize={t: float(price_day_ahead[t]) for t in range(T)})
        
        # E-programma parameters (voorspelling gebaseerd op load/PV zonder batterij)
        netpos_without_battery = load - pv
        e_program_factor = e_program_percentage / 100.0
        model.e_program = Param(model.T, initialize={t: float(netpos_without_battery[t] * e_program_factor) for t in range(T)})
        
        def netpos_def(model, t):
            return model.netpos[t] == model.load_profile[t] - model.pv[t] + (model.charge[t] - model.discharge[t]) * time_step_h
        model.netpos_con = Constraint(model.T, rule=netpos_def)
        
        def netpos_split(model, t):
            return model.netpos[t] == model.netpos_pos[t] - model.netpos_neg[t]
        model.netpos_split_con = Constraint(model.T, rule=netpos_split)
        
        # Onbalans berekening: verschil tussen werkelijke netpos en e-programma
        def imbalance_def(model, t):
            return model.imbalance[t] == model.netpos[t] - model.e_program[t]
        model.imbalance_con = Constraint(model.T, rule=imbalance_def)
        
        def imbalance_split(model, t):
            return model.imbalance[t] == model.imbalance_pos[t] - model.imbalance_neg[t]
        model.imbalance_split_con = Constraint(model.T, rule=imbalance_split)
        
        # Objective: minimaliseer totale kosten (day-ahead kosten voor e-programma + onbalanskosten voor afwijkingen)
        def objective(model):
            # Day-ahead kosten voor e-programma (positief = afname = kosten, negatief = invoeding = opbrengst)
            day_ahead_term = sum(
                model.price_day_ahead[t] * model.e_program[t] for t in model.T
            )
            # Onbalanskosten voor afwijkingen (positief = shortage kosten, negatief = surplus opbrengst)
            imbalance_term = sum(
                model.price_shortage[t] * model.imbalance_pos[t] - model.price_surplus[t] * model.imbalance_neg[t]
                for t in model.T
            )
            return day_ahead_term + imbalance_term
        model.obj = Objective(rule=objective, sense=minimize)

        # Eind-SoC constraint
        model.final_soc_min = Constraint(expr=model.soc[T-1] + (
            model.charge[T-1] * time_step_h * eff_ch - model.discharge[T-1] * time_step_h * eff_dis) >= min_soc)
        model.final_soc_max = Constraint(expr=model.soc[T-1] + (
            model.charge[T-1] * time_step_h * eff_ch - model.discharge[T-1] * time_step_h * eff_dis) <= max_soc)

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
            soc_reset_attempted = False
            
            # Simuleer wat er zou gebeuren zonder SoC constraints
            try:
                # Probeer een vereenvoudigde berekening om te zien of SoC het probleem is
                test_soc = current_soc
                daily_charge_total = 0
                daily_discharge_total = 0
                
                # Schat minimale acties voor verplichte laden/ontladen
                for t in range(T):
                    space_ch = day_data["space available for charging (kWh)"].iloc[t]
                    space_dis = day_data["space available for discharging (kWh)"].iloc[t]
                    
                    min_discharge = 0
                    min_charge = 0
                    
                    if space_ch < 0:  # Verplicht ontladen
                        min_discharge = abs(space_ch) / 0.25 / 1000  # MW
                        daily_discharge_total += min_discharge * time_step_h
                        test_soc -= min_discharge * time_step_h / eff_dis
                    
                    if space_dis < 0:  # Verplicht laden
                        min_charge = abs(space_dis) / 0.25 / 1000  # MW
                        daily_charge_total += min_charge * time_step_h
                        test_soc += min_charge * time_step_h * eff_ch
                
                # Check of SoC buiten tolerantie zou gaan
                if test_soc < (min_soc - soc_tolerance) or test_soc > (max_soc + soc_tolerance):
                    soc_violation_detected = False  # Te grote overschrijding
                elif test_soc < min_soc or test_soc > max_soc:
                    soc_violation_detected = True  # Kleine overschrijding die we kunnen tolereren
                
            except:
                soc_violation_detected = False
            
            # Als het een tolereerbare SoC overschrijding is, reset SoC en probeer opnieuw
            if soc_violation_detected:
                soc_reset_attempted = True
                
                # Bepaal of we naar min of max SoC moeten resetten
                if test_soc < min_soc:
                    # SoC zou te laag worden, reset naar minimum SoC
                    reset_soc = min_soc
                    reset_direction = "minimum"
                else:
                    # SoC zou te hoog worden, reset naar maximum SoC  
                    reset_soc = max_soc
                    reset_direction = "maximum"
                
                # Update current_soc voor deze dag
                original_soc = current_soc
                current_soc = reset_soc
                
                # Log de SoC reset
                warning_msg = f"SoC reset: Dag {day.strftime('%d-%m-%Y')} - batterij SoC gereset naar {reset_direction} ({reset_soc:.3f} MWh, was {original_soc:.3f} MWh)"
                if progress_callback:
                    progress_callback(warning_msg)
                
                # Probeer de optimalisatie opnieuw met de gereset SoC
                # Update het model met de nieuwe start SoC
                model.soc[0].fix(current_soc)
                
                # Solve opnieuw
                result = solver.solve(model)
                
                # Check opnieuw voor infeasibility
                if (result.solver.status != SolverStatus.ok) or (result.solver.termination_condition == TerminationCondition.infeasible):
                    # Als het nog steeds niet lukt, ga door naar de normale infeasible handling
                    pass  # Ga door naar de normale infeasible dagen behandeling hieronder
                else:
                    # Het lukte nu wel! Ga door met normale verwerking
                    charge = [model.charge[t]() for t in model.T]
                    discharge = [model.discharge[t]() for t in model.T]
                    soc = [model.soc[t]() for t in model.T]

                    # Update SoC voor volgende dag
                    final_charge = charge[-1]
                    final_discharge = discharge[-1] 
                    final_soc = soc[-1] + (final_charge * time_step_h * eff_ch - final_discharge * time_step_h / eff_dis)
                    final_soc = min(max(final_soc, min_soc), max_soc)
                    current_soc = final_soc

                    # Update cycles en budget
                    charged_energy = sum(charge) * time_step_h  # in MWh
                    daily_cycle = (charged_energy * eff_ch) / usable_capacity
                    cumulative_cycles += daily_cycle
                    remaining_cycles -= daily_cycle
                    remaining_cycles = max(0, remaining_cycles)
                    cycle_history.append(daily_cycle)

                    # Netto netpositie en opbrengst/kosten per timestep
                    netpos = load - pv + (np.array(charge) - np.array(discharge)) * time_step_h
                    
                    # Bereken e-programma en onbalans
                    netpos_without_battery = load - pv
                    e_program_netpos = netpos_without_battery * (e_program_percentage / 100.0)
                    imbalance = netpos - e_program_netpos
                    
                    # Day-ahead kosten voor e-programma
                    price_day_ahead = day_data["price_day_ahead"].values
                    day_ahead_costs = np.where(e_program_netpos > 0, -price_day_ahead * e_program_netpos, price_day_ahead * abs(e_program_netpos))
                    
                    # Onbalanskosten voor afwijkingen
                    imbalance_costs = np.where(imbalance > 0, -price_shortage * imbalance, price_surplus * abs(imbalance))
                    
                    # Totale kosten
                    opbrengst_kosten = day_ahead_costs + imbalance_costs

                    # Prepare columns for output
                    soc_kwh = [s * 1000 for s in soc]  # SoC in kWh
                    day_data = day_data.copy()
                    if 'space available for charging (kWh)' in day_data.columns:
                        day_data['space_for_charging_kWh'] = day_data['space available for charging (kWh)']
                        day_data.drop(columns=['space available for charging (kWh)'], inplace=True)
                    if 'space available for discharging (kWh)' in day_data.columns:
                        day_data['space_for_discharging_kWh'] = day_data['space available for discharging (kWh)']
                        day_data.drop(columns=['space available for discharging (kWh)'], inplace=True)
                    day_data['energy_charged_kWh'] = np.array(charge) * time_step_h * 1000
                    day_data['energy_discharged_kWh'] = np.array(discharge) * time_step_h * 1000
                    day_data['SoC_kWh'] = soc_kwh
                    day_data['SoC_pct'] = [(s - min_soc) / (max_soc - min_soc) for s in soc]
                    day_data['grid_exchange_kWh'] = netpos * 1000  # Converteer naar kWh voor export
                    day_data['e_program_kWh'] = e_program_netpos * 1000  # Converteer naar kWh voor export
                    day_data['day_ahead_result'] = day_ahead_costs  # Kan positief (inkomsten) of negatief (kosten) zijn
                    day_data['imbalance_result'] = imbalance_costs  # Kan positief (inkomsten) of negatief (kosten) zijn
                    
                    # Bereken energy_tax en supplier_costs voor SoC reset dagen
                    if 'load' in day_data.columns and 'production_PV' in day_data.columns:
                        net_grid_consumption_kwh = (day_data['load'].values 
                                                   - day_data['production_PV'].values 
                                                   + np.array(charge) * time_step_h * 1000
                                                   - np.array(discharge) * time_step_h * 1000)
                        net_grid_consumption_mwh = net_grid_consumption_kwh / 1000
                        
                        energy_tax_costs = np.where(net_grid_consumption_mwh > 0, -net_grid_consumption_mwh * marginal_tax_rate, 0)
                        supplier_cost_values = -np.abs(net_grid_consumption_mwh) * supply_costs
                        
                        day_data['energy_tax'] = energy_tax_costs
                        day_data['supplier_costs'] = supplier_cost_values
                    else:
                        day_data['energy_tax'] = 0
                        day_data['supplier_costs'] = 0
                    day_data['transport_costs'] = -np.array(transport_costs_per_timestep)  # Altijd negatief (kosten)
                    
                    # Total result = day-ahead + imbalance + alle extra kosten (voor SoC reset dagen)
                    total_result = (opbrengst_kosten + 
                                   day_data['energy_tax'] + 
                                   day_data['supplier_costs'] + 
                                   day_data['transport_costs'])
                    day_data['total_result_imbalance_PAP'] = total_result
                    results.append(day_data)
                    
                    # Ga door naar de volgende dag
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
            
            # Voor echt infeasible dagen: bepaal juiste SoC en batterij doet niets
            reset_soc_used = current_soc
            reset_reason = "onbekend"
            
            # Als we al een SoC reset hebben gedaan, behoud die waarde
            if soc_reset_attempted:
                # We hadden al een reset gedaan, dus behoud die
                if current_soc == min_soc:
                    reset_reason = "minimum (na eerdere reset)"
                elif current_soc == max_soc:
                    reset_reason = "maximum (na eerdere reset)"
                else:
                    reset_reason = f"gereset waarde ({current_soc:.3f} MWh)"
            else:
                # Geen eerdere reset, bepaal beste SoC gebaseerd op het probleem
                if current_soc <= min_soc + 0.01:  # Bijna op minimum
                    reset_soc_used = min_soc
                    reset_reason = "minimum (batterij was bijna leeg)"
                elif current_soc >= max_soc - 0.01:  # Bijna op maximum  
                    reset_soc_used = max_soc
                    reset_reason = "maximum (batterij was bijna vol)"
                else:
                    # Anders behoud huidige SoC
                    reset_reason = f"huidige waarde ({current_soc:.3f} MWh)"
            
            current_soc = reset_soc_used
            warning_msg_reset = f"Infeasible dag: {dag_datum} - batterij SoC op {reset_reason}, batterij inactief"
            if progress_callback:
                progress_callback(warning_msg_reset)
            
            # Batterij doet niets deze dag (blijft op reset_soc_used)
            no_action_charge = [0.0] * T
            no_action_discharge = [0.0] * T
            no_action_soc = [reset_soc_used] * T
            
            # Bereken net position zonder batterij  
            load = day_data["load"].values / 1000  # MWh
            pv = day_data["production_PV"].values / 1000  # MWh
            netpos = load - pv  # Geen batterij bijdrage
            
            # Bereken e-programma en onbalans voor infeasible dagen
            netpos_without_battery = load - pv
            e_program_netpos = netpos_without_battery * (e_program_percentage / 100.0)
            imbalance = netpos - e_program_netpos
            
            price_shortage = day_data["price_shortage"].values
            price_surplus = day_data["price_surplus"].values
            price_day_ahead = day_data["price_day_ahead"].values
            
            # Day-ahead kosten voor e-programma
            day_ahead_costs = np.where(e_program_netpos > 0, -price_day_ahead * e_program_netpos, price_day_ahead * abs(e_program_netpos))
            
            # Onbalanskosten voor afwijkingen
            imbalance_costs = np.where(imbalance > 0, -price_shortage * imbalance, price_surplus * abs(imbalance))
            
            # Totale kosten
            opbrengst_kosten = day_ahead_costs + imbalance_costs
            
            # Voeg resultaten toe
            day_data = day_data.copy()
            if 'space available for charging (kWh)' in day_data.columns:
                day_data['space_for_charging_kWh'] = day_data['space available for charging (kWh)']
                day_data.drop(columns=['space available for charging (kWh)'], inplace=True)
            if 'space available for discharging (kWh)' in day_data.columns:
                day_data['space_for_discharging_kWh'] = day_data['space available for discharging (kWh)']
                day_data.drop(columns=['space available for discharging (kWh)'], inplace=True)
            day_data['energy_charged_kWh'] = [c * time_step_h * 1000 for c in no_action_charge]
            day_data['energy_discharged_kWh'] = [d * time_step_h * 1000 for d in no_action_discharge]
            day_data['SoC_kWh'] = [s * 1000 for s in no_action_soc]
            day_data['SoC_pct'] = [(s - min_soc) / (max_soc - min_soc) for s in no_action_soc]
            day_data['grid_exchange_kWh'] = netpos * 1000  # Converteer naar kWh voor export
            day_data['e_program_kWh'] = e_program_netpos * 1000  # Converteer naar kWh voor export
            day_data['day_ahead_result'] = day_ahead_costs  # Kan positief (inkomsten) of negatief (kosten) zijn
            day_data['imbalance_result'] = imbalance_costs  # Kan positief (inkomsten) of negatief (kosten) zijn
            
            # Bereken energy_tax en supplier_costs voor infeasible dagen (zonder batterij)
            if 'load' in day_data.columns and 'production_PV' in day_data.columns:
                net_grid_consumption_kwh = day_data['load'].values - day_data['production_PV'].values  # Zonder batterij
                net_grid_consumption_mwh = net_grid_consumption_kwh / 1000
                
                energy_tax_costs = np.where(net_grid_consumption_mwh > 0, -net_grid_consumption_mwh * marginal_tax_rate, 0)
                supplier_cost_values = -np.abs(net_grid_consumption_mwh) * supply_costs
                
                day_data['energy_tax'] = energy_tax_costs
                day_data['supplier_costs'] = supplier_cost_values
            else:
                day_data['energy_tax'] = 0
                day_data['supplier_costs'] = 0
            day_data['transport_costs'] = -np.array(transport_costs_per_timestep)  # Altijd negatief (kosten)
            
            # Total result = day-ahead + imbalance + alle extra kosten (voor infeasible dagen)
            total_result = (opbrengst_kosten + 
                           day_data['energy_tax'] + 
                           day_data['supplier_costs'] + 
                           day_data['transport_costs'])
            day_data['total_result_imbalance_PAP'] = total_result
            results.append(day_data)
            
            # Ga door naar de volgende dag
            continue

        charge = [model.charge[t]() for t in model.T]
        discharge = [model.discharge[t]() for t in model.T]
        soc = [model.soc[t]() for t in model.T]

        # Update SoC voor volgende dag
        final_charge = charge[-1]
        final_discharge = discharge[-1] 
        final_soc = soc[-1] + (final_charge * time_step_h * eff_ch - final_discharge * time_step_h / eff_dis)
        final_soc = min(max(final_soc, min_soc), max_soc)
        current_soc = final_soc

        # Update cycles en budget
        charged_energy = sum(charge) * time_step_h  # in MWh
        daily_cycle = (charged_energy * eff_ch) / usable_capacity
        cumulative_cycles += daily_cycle
        remaining_cycles -= daily_cycle
        remaining_cycles = max(0, remaining_cycles)
        cycle_history.append(daily_cycle)

        # Netto netpositie en opbrengst/kosten per timestep
        netpos = load - pv + (np.array(charge) - np.array(discharge)) * time_step_h
        
        # Bereken e-programma (voorspelling) gebaseerd op de werkelijke load/PV zonder batterij
        # E-programma is percentage van werkelijke afname/invoeding
        netpos_without_battery = load - pv  # Zonder batterij invloed
        e_program_netpos = netpos_without_battery * (e_program_percentage / 100.0)
        
        # Bereken onbalans: verschil tussen werkelijke netpositie en e-programma
        imbalance = netpos - e_program_netpos
        
        # Bereken day-ahead kosten voor het e-programma
        price_day_ahead = day_data["price_day_ahead"].values
        day_ahead_costs = np.where(e_program_netpos > 0, -price_day_ahead * e_program_netpos, price_day_ahead * abs(e_program_netpos))
        
        # Onbalanskosten/-opbrengsten berekenen voor afwijkingen van e-programma:
        # Positieve onbalans (meer afgenomen dan voorspeld) -> shortage prijs betalen
        # Negatieve onbalans (meer ingevoed dan voorspeld) -> surplus prijs ontvangen
        imbalance_costs = np.where(imbalance > 0, -price_shortage * imbalance, price_surplus * abs(imbalance))
        
        # Totale kosten = day-ahead kosten voor e-programma + onbalanskosten voor afwijkingen
        opbrengst_kosten = day_ahead_costs + imbalance_costs

        # Bereken transportkosten: alleen voor netto afname van het net
        # Voor imbalance_everything_PAP: bereken op basis van netpos (netto positie na batterij)
        # Transportkosten alleen voor positieve netpos (netto afname)
        transport_costs_per_timestep = [max(0, net_pos) * transport_costs for net_pos in netpos]  # MWh * €/MWh
        
        # Prepare columns for output
        soc_kwh = [s * 1000 for s in soc]  # SoC in kWh
        day_data = day_data.copy()
        if 'space available for charging (kWh)' in day_data.columns:
            day_data['space_for_charging_kWh'] = day_data['space available for charging (kWh)']
            day_data.drop(columns=['space available for charging (kWh)'], inplace=True)
        if 'space available for discharging (kWh)' in day_data.columns:
            day_data['space_for_discharging_kWh'] = day_data['space available for discharging (kWh)']
            day_data.drop(columns=['space available for discharging (kWh)'], inplace=True)
        day_data['energy_charged_kWh'] = np.array(charge) * time_step_h * 1000
        day_data['energy_discharged_kWh'] = np.array(discharge) * time_step_h * 1000
        day_data['SoC_kWh'] = soc_kwh
        day_data['SoC_pct'] = [(s - min_soc) / (max_soc - min_soc) for s in soc]
        day_data['grid_exchange_kWh'] = netpos * 1000  # Converteer naar kWh voor export
        day_data['e_program_kWh'] = e_program_netpos * 1000  # Converteer naar kWh voor export
        day_data['day_ahead_result'] = day_ahead_costs  # Kan positief (inkomsten) of negatief (kosten) zijn
        day_data['imbalance_result'] = imbalance_costs  # Kan positief (inkomsten) of negatief (kosten) zijn
        
        # Bereken energy_tax en supplier_costs gebaseerd op netto grid verbruik
        # Netto grid verbruik = load - PV + batterij laden - batterij ontladen (zoals in day-ahead algoritme)
        if 'load' in day_data.columns and 'production_PV' in day_data.columns:
            net_grid_consumption_kwh = (day_data['load'].values 
                                       - day_data['production_PV'].values 
                                       + np.array(charge) * time_step_h * 1000      # Laden = extra afname (kWh)
                                       - np.array(discharge) * time_step_h * 1000)  # Ontladen = minder afname (kWh)
            net_grid_consumption_mwh = net_grid_consumption_kwh / 1000  # Converteer naar MWh
            
            # Energiebelasting alleen voor positief verbruik (afname van net) - altijd negatief (kosten)
            energy_tax_costs = np.where(net_grid_consumption_mwh > 0, -net_grid_consumption_mwh * marginal_tax_rate, 0)
            # Leveringskosten gelden voor absolute waarde van afname EN invoeding - altijd negatief (kosten)
            supplier_cost_values = -np.abs(net_grid_consumption_mwh) * supply_costs
            
            day_data['energy_tax'] = energy_tax_costs
            day_data['supplier_costs'] = supplier_cost_values
        else:
            day_data['energy_tax'] = 0  # Geen load/PV data beschikbaar
            day_data['supplier_costs'] = 0
        day_data['transport_costs'] = -np.array(transport_costs_per_timestep)  # Altijd negatief (kosten)
        
        # Total result = day-ahead + imbalance + alle extra kosten
        total_result = (opbrengst_kosten + 
                       day_data['energy_tax'] + 
                       day_data['supplier_costs'] + 
                       day_data['transport_costs'])
        day_data['total_result_imbalance_PAP'] = total_result
        results.append(day_data)

    final_df = pd.concat(results)
    total_result = final_df['total_result_imbalance_PAP'].sum()
    total_cycles = cumulative_cycles

    return final_df, {
        "total_result": total_result,
        "total_cycles": total_cycles,
        "cycle_history": cycle_history,
        "battery_power_MW": power_mw,
        "infeasible_days": infeasible_days
    }


