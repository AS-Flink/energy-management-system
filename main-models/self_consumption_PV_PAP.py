import pandas as pd
import numpy as np
import os
from pyomo.environ import *
from pyomo.opt import SolverFactory

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
    return tax_table['consumption_brackets'][-1]['tax_eur_per_mwh']

def run_battery_trading(config, progress_callback=None):
    # Read Excel sheet
    df = config.input_data.copy()
    datetime_col = None
    for col in df.columns:
        if col.strip().lower() == 'datetime':
            datetime_col = col
            break
    if not datetime_col:
        raise ValueError("No column 'Datetime' of 'datetime' gevonden in het 'Export naar Python' sheet.")
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df.set_index(datetime_col, inplace=True)

    # Check of benodigde kolommen aanwezig zijn
    required_columns = ["production_PV", "load", "price_day_ahead", "space available for charging (kWh)", "space available for discharging (kWh)", 
                       "grid_excl_battery", "max_feed_in_grid", "max_take_from_grid"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' ontbreekt in de input data.")

    # Configuratie
    power_mw = config.POWER_MW
    capacity_mwh = config.CAPACITY_MWH
    eff_ch = config.EFF_CH
    eff_dis = config.EFF_DIS
    min_soc_frac = config.MIN_SOC
    max_soc_frac = config.MAX_SOC
    min_soc = min_soc_frac * capacity_mwh
    max_soc = max_soc_frac * capacity_mwh
    time_step_h = config.TIME_STEP_H
    max_cycles = config.MAX_CYCLES
    if hasattr(config, 'INIT_SOC'):
        start_soc = float(config.INIT_SOC) * capacity_mwh  # We bepalen dit later per dag
    else:
        start_soc = 0.5 * capacity_mwh
    usable_capacity = capacity_mwh * (max_soc_frac - min_soc_frac)
    
    # Energiebelasting tabel en leveringskosten
    tax_table = get_energy_tax_table()
    
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
    
    # Schat jaarverbruik voor belastingberekening (simpele benadering)
    if len(df) > 0:
        # Bereken gemiddeld verbruik per tijdstap en extrapoleer naar jaar
        avg_load_kwh = df['load'].mean()
        hours_per_year = 8760
        estimated_annual_consumption_mwh = (avg_load_kwh * hours_per_year) / 1000
        
        # Bereken marginale energiebelasting
        marginal_tax_rate = calculate_energy_tax(estimated_annual_consumption_mwh, tax_table)
        
        if progress_callback:
            progress_callback(f"Geschat jaarverbruik: {estimated_annual_consumption_mwh:.1f} MWh")
            progress_callback(f"Energiebelasting: €{marginal_tax_rate:.2f}/MWh")
            progress_callback(f"Leveringskosten: €{supply_costs:.2f}/MWh (voor afname en invoeding)")
            progress_callback(f"Transportkosten: €{transport_costs:.2f}/MWh (alleen voor afname)")
    else:
        marginal_tax_rate = tax_table['consumption_brackets'][0]['tax_eur_per_mwh']

    # Converteer input data van kWh naar MWh voor Pyomo
    df_mw = df.copy()
    # Converteer alle energie kolommen van kWh naar MWh
    df_mw['production_PV_mwh'] = df['production_PV'] / 1000  # kWh -> MWh
    df_mw['load_mwh'] = df['load'] / 1000  # kWh -> MWh
    df_mw['space_charging_mwh'] = df['space available for charging (kWh)'] / 1000  # kWh -> MWh
    df_mw['space_discharging_mwh'] = df['space available for discharging (kWh)'] / 1000  # kWh -> MWh
    df_mw['grid_excl_battery_mwh'] = df['grid_excl_battery'] / 1000  # kWh -> MWh
    df_mw['max_feed_in_grid_mwh'] = df['max_feed_in_grid'] / 1000  # kWh -> MWh
    df_mw['max_take_from_grid_mwh'] = df['max_take_from_grid'] / 1000  # kWh -> MWh
    
    # Initialisatie
    network_violations = []
    
    # Maak Pyomo model voor jaar-optimalisatie
    if progress_callback:
        progress_callback("Pyomo model opbouwen...")
    
    # Maak tijdstappen index
    timesteps = list(range(len(df)))
    
    # Maak Pyomo model
    model = ConcreteModel()
    
    # Variabelen - Pure lineaire variabelen (geen binaire) - ALLES IN MW/MWh
    model.charge = Var(timesteps, domain=NonNegativeReals, bounds=(0, power_mw))  # MW
    model.discharge = Var(timesteps, domain=NonNegativeReals, bounds=(0, power_mw))  # MW
    model.soc = Var(timesteps, domain=NonNegativeReals, bounds=(min_soc, max_soc))  # MWh
    model.feed_in_violation = Var(timesteps, domain=NonNegativeReals, bounds=(0, 1.0))  # MWh - slack voor feed-in overschrijding
    model.take_from_violation = Var(timesteps, domain=NonNegativeReals, bounds=(0, 1.0))  # MWh - slack voor take-from overschrijding
    model.grid_feed_in = Var(timesteps, domain=NonNegativeReals, bounds=(0, 1.0))  # MWh - hoeveelheid grid feed-in per tijdstap
    
    # Fix initial SoC
    current_soc = start_soc
    model.soc[0].fix(current_soc)
    
    # Linear constraint: sum of charge + discharge can't exceed max power
    def mutual_exclusion_rule(model, t):
        return model.charge[t] + model.discharge[t] <= power_mw
    model.mutual_exclusion = Constraint(timesteps, rule=mutual_exclusion_rule)
    
    # SoC balans
    def soc_balance_rule(model, t):
        if t == 0: 
            return Constraint.Skip  # Eerste tijdstap is gefixed
        return model.soc[t] == model.soc[t-1] + (
            model.charge[t-1] * time_step_h * eff_ch -
            model.discharge[t-1] * time_step_h / eff_dis)
    
    model.soc_balance = Constraint(timesteps, rule=soc_balance_rule)
    
    # Netwerk grenzen met slack variabelen - ALLES IN MWh
    def network_feed_in_rule(model, t):
        grid_excl = df_mw.iloc[t]['grid_excl_battery_mwh']  # MWh
        max_feed_in = df_mw.iloc[t]['max_feed_in_grid_mwh']  # MWh
        net_battery_exchange = (model.charge[t] - model.discharge[t]) * time_step_h  # MWh (positief=laden, negatief=ontladen)
        grid_incl = grid_excl + net_battery_exchange
        return grid_incl >= max_feed_in - model.feed_in_violation[t]
    
    def network_take_from_rule(model, t):
        grid_excl = df_mw.iloc[t]['grid_excl_battery_mwh']  # MWh
        max_take_from = df_mw.iloc[t]['max_take_from_grid_mwh']  # MWh
        net_battery_exchange = (model.charge[t] - model.discharge[t]) * time_step_h  # MWh (positief=laden, negatief=ontladen)
        grid_incl = grid_excl + net_battery_exchange
        return grid_incl <= max_take_from + model.take_from_violation[t]
    
    model.network_feed_in = Constraint(timesteps, rule=network_feed_in_rule)
    model.network_take_from = Constraint(timesteps, rule=network_take_from_rule)
    
    # Grid feed-in constraint - ALLES IN MWh
    def grid_feed_in_rule(model, t):
        grid_excl = df_mw.iloc[t]['grid_excl_battery_mwh']  # MWh
        net_battery_exchange = (model.charge[t] - model.discharge[t]) * time_step_h  # MWh
        grid_incl = grid_excl + net_battery_exchange
        return model.grid_feed_in[t] >= -grid_incl
    
    model.grid_feed_in_constraint = Constraint(timesteps, rule=grid_feed_in_rule)
    
    # Cycli limiet
    if max_cycles > 0 and usable_capacity > 0:
        total_charged_energy = sum(model.charge[t] * time_step_h for t in timesteps)  # MWh
        max_energy_allowed = max_cycles * usable_capacity  # MWh
        if progress_callback:
            progress_callback(f"Cycle constraint: max {max_energy_allowed:.2f} MWh geladen per jaar")
        model.cycle_limit = Constraint(expr=total_charged_energy * eff_ch <= max_energy_allowed)
    
    # Doel: minimaliseer netwerkoverschrijdingen en grid feed-in, maximaliseer zelfconsumptie - ALLES IN MW/MWh
    def objective_rule(model):
        # Straf netwerkoverschrijdingen zwaar (hoogste prioriteit) - MWh * 100000 voor juiste schaling
        violation_penalty = sum(model.feed_in_violation[t] + model.take_from_violation[t] for t in timesteps) * 100000
        
        # Straf grid feed-in (PV-stroom liever opslaan dan terugleveren) - MWh * 1000 voor juiste schaling
        grid_feed_in_penalty = sum(model.grid_feed_in[t] for t in timesteps) * 1000
        
        # Lineaire straf op totale activiteit (stimuleert efficiënt gebruik, voorkomt simultaan) - MW * 100 voor juiste schaling
        total_activity_penalty = sum(model.charge[t] + model.discharge[t] for t in timesteps) * 1
        
        # Beloning voor laden bij PV overschot
        pv_self_consumption_reward = 0
        for t in timesteps:
            if df_mw.iloc[t]['grid_excl_battery_mwh'] < 0:  # Er is PV netlevering (MWh)
                pv_self_consumption_reward -= model.charge[t] * time_step_h * 500  # Beloning - MWh * 10 voor juiste schaling
        
        return violation_penalty + grid_feed_in_penalty + total_activity_penalty + pv_self_consumption_reward
    
    model.objective = Objective(rule=objective_rule, sense=minimize)
    
    # Los het model op
    if progress_callback:
        progress_callback("Pyomo optimalisatie uitvoeren...")
    
    # Probeer eerst de lokale CBC solver
    cbc_path = os.path.join(os.path.dirname(__file__), 'Cbc-releases.2.10.12-w64-msvc16-md', 'bin', 'cbc.exe')
    solver = None
    
    if os.path.exists(cbc_path):
        try:
            solver = SolverFactory('cbc', executable=cbc_path)
            if progress_callback:
                progress_callback(f"Lokale CBC solver gevonden: {cbc_path}")
        except Exception as e:
            if progress_callback:
                progress_callback(f"Fout bij laden lokale CBC: {e}")
    
    if solver is None:
        try:
            solver = SolverFactory('cbc')
            if progress_callback:
                progress_callback("Standaard CBC solver gebruikt")
        except Exception as e:
            if progress_callback:
                progress_callback(f"Fout bij laden standaard CBC: {e}")
            raise ValueError("Geen CBC solver beschikbaar")
    
    try:
        # Configureer solver voor lineair probleem (LP)
        solver.options['presolve'] = 'on'
        solver.options['scaling'] = 'on'
        solver.options['primalT'] = 1e-6
        solver.options['dualT'] = 1e-6
        solver.options['timeLimit'] = 300  # 5 minuten timeout
        
        if progress_callback:
            progress_callback(f"Model statistieken: {len(timesteps)} tijdstappen")
        
        results_pyomo = solver.solve(model, tee=False)
        
        if progress_callback and results_pyomo:
            progress_callback(f"Solver status: {results_pyomo.solver.status}")
            progress_callback(f"Termination condition: {results_pyomo.solver.termination_condition}")
    except Exception as e:
        if progress_callback:
            progress_callback(f"CBC solver fout: {str(e)}. Gebruik fallback heuristiek...")
        results_pyomo = None
    
    # Accepteer optimale en feasible oplossingen
    acceptable_conditions = [
        TerminationCondition.optimal,
        TerminationCondition.feasible,
        TerminationCondition.maxTimeLimit,
        TerminationCondition.maxIterations,
        TerminationCondition.other,
        TerminationCondition.userInterrupt
    ]
    
    if results_pyomo and results_pyomo.solver.termination_condition in acceptable_conditions:
        if progress_callback:
            progress_callback("Pyomo optimalisatie succesvol! Resultaten verwerken...")
        
        # Haal resultaten op (in MW/MWh)
        charge_list = [value(model.charge[t]) for t in timesteps]  # MW
        discharge_list = [value(model.discharge[t]) for t in timesteps]  # MW
        soc_list = [value(model.soc[t]) for t in timesteps]  # MWh
        feed_in_violations = [value(model.feed_in_violation[t]) for t in timesteps]  # MWh
        take_from_violations = [value(model.take_from_violation[t]) for t in timesteps]  # MWh
        
        # Converteer resultaten terug naar kWh voor output compatibiliteit
        energy_charged_list = [charge_list[i] * time_step_h * 1000 for i in range(len(timesteps))]  # MW * h * 1000 = kWh
        energy_discharged_list = [discharge_list[i] * time_step_h * 1000 for i in range(len(timesteps))]  # MW * h * 1000 = kWh
        soc_frac_list = [(soc_list[i] - min_soc) / (max_soc - min_soc) for i in range(len(timesteps))]
        
        # Controleer op netwerkoverschrijdingen (converteer terug naar kWh voor rapportage)
        for i, t in enumerate(timesteps):
            if feed_in_violations[i] > 0.000001:  # MWh threshold
                violation = {
                    'datetime': df.index[i],
                    'type': 'feed_in',
                    'max_allowed': abs(df.iloc[i]['max_feed_in_grid']),  # kWh
                    'actual': abs(df.iloc[i]['grid_excl_battery'] + (energy_charged_list[i] - energy_discharged_list[i])),  # kWh
                    'violation': feed_in_violations[i] * 1000  # MWh -> kWh
                }
                network_violations.append(violation)
            
            if take_from_violations[i] > 0.000001:  # MWh threshold
                violation = {
                    'datetime': df.index[i],
                    'type': 'take_from',
                    'max_allowed': df.iloc[i]['max_take_from_grid'],  # kWh
                    'actual': df.iloc[i]['grid_excl_battery'] + (energy_charged_list[i] - energy_discharged_list[i]),  # kWh
                    'violation': take_from_violations[i] * 1000  # MWh -> kWh
                }
                network_violations.append(violation)
        
        # Bereken cycli (energy_charged_list is al in kWh, dus converteren naar MWh)
        total_charged_energy = sum(energy_charged_list) / 1000  # kWh -> MWh
        total_cycles = (total_charged_energy * eff_ch) / usable_capacity if usable_capacity > 0 else 0
        
        # Maak resultaat DataFrame
        final_df = df.copy()
        final_df['energy_charged_kWh'] = energy_charged_list
        final_df['energy_discharged_kWh'] = energy_discharged_list
        final_df['SoC_kWh'] = [s * 1000 for s in soc_list]
        final_df['SoC_pct'] = soc_frac_list
        
        # Voeg ontbrekende kolommen toe zoals in de fallback functie
        # Voeg grid_exchange kolom toe - toont werkelijke afname/invoeding van net na batterij
        if 'grid_exchange_kWh' not in final_df.columns:
            # Bereken net uitwisseling met grid: positief = afname van net, negatief = invoeding naar net
            final_df['grid_exchange_kWh'] = (final_df['load'] 
                                             - final_df['production_PV'] 
                                             + final_df['energy_charged_kWh']      # Laden = extra afname
                                             - final_df['energy_discharged_kWh'])  # Ontladen = minder afname
        
        # Voeg space kolommen toe indien nog niet aanwezig
        if 'space_for_charging_kWh' not in final_df.columns and 'space available for charging (kWh)' in final_df.columns:
            final_df['space_for_charging_kWh'] = final_df['space available for charging (kWh)']
        if 'space_for_discharging_kWh' not in final_df.columns and 'space available for discharging (kWh)' in final_df.columns:
            final_df['space_for_discharging_kWh'] = final_df['space available for discharging (kWh)']
        
        # Voeg uitgebreide kostenberekening toe (identiek aan day_ahead_trading_PAP)
        if 'total_result_self_consumption' not in final_df.columns:
            # Bereken totale energiekosten inclusief belasting en netkosten
            if 'price_day_ahead' in final_df.columns:
                # Netto grid verbruik = load - PV productie + batterij laden - batterij ontladen
                net_grid_consumption_kwh = (final_df['load'] 
                                           - final_df['production_PV'] 
                                           + final_df['energy_charged_kWh']      # Laden = extra inkoop (kWh)
                                           - final_df['energy_discharged_kWh'])  # Ontladen = minder inkoop of verkoop (kWh)
                
                net_grid_consumption_mwh = net_grid_consumption_kwh / 1000  # Converteer naar MWh
                
                # Bereken kosten per component
                day_ahead_costs = net_grid_consumption_mwh * final_df['price_day_ahead']
                
                # Voor positief verbruik (afname): voeg belasting, leveringskosten en transportkosten toe
                # Voor negatief verbruik (invoeding): alleen leveringskosten (energiebelasting en transportkosten niet van toepassing)
                tax_costs = np.where(net_grid_consumption_mwh > 0, net_grid_consumption_mwh * marginal_tax_rate, 0)
                
                # Transportkosten alleen voor netto afname van het net
                transport_costs_all = np.where(net_grid_consumption_mwh > 0, net_grid_consumption_mwh * transport_costs, 0)
                
                # Leveringskosten gelden voor absolute waarde van afname EN invoeding
                supply_costs_all = np.abs(net_grid_consumption_mwh) * supply_costs
                
                additional_costs = tax_costs + transport_costs_all + supply_costs_all
                
                # Voeg detail kolommen toe voor transparantie - EERST de kolommen aanmaken
                final_df['dummy1'] = 0
                final_df['dummy2'] = 0
                final_df['day_ahead_result'] = day_ahead_costs  # Kan positief (inkomsten) of negatief (kosten) zijn
                final_df['dummy3'] = 0
                final_df['energy_tax'] = np.where(net_grid_consumption_mwh > 0, -net_grid_consumption_mwh * marginal_tax_rate, 0)  # Altijd negatief (kosten)
                final_df['supplier_costs'] = -np.abs(net_grid_consumption_mwh) * supply_costs  # Altijd negatief (kosten)
                final_df['transport_costs'] = np.where(net_grid_consumption_mwh > 0, -net_grid_consumption_mwh * transport_costs, 0)  # Altijd negatief (kosten)
                
                # Totale kosten = day-ahead + alle extra kosten (nu met correcte voortekens)
                final_df['total_result_self_consumption'] = (day_ahead_costs + 
                                                             final_df['energy_tax'] + 
                                                             final_df['supplier_costs'] + 
                                                             final_df['transport_costs'])
            else:
                final_df['total_result_self_consumption'] = 0
                final_df['dummy1'] = 0
                final_df['dummy2'] = 0
                final_df['day_ahead_result'] = 0
                final_df['dummy3'] = 0
                final_df['energy_tax'] = 0
                final_df['supplier_costs'] = 0
                final_df['transport_costs'] = 0
        
        # Voeg grid_exchange_day_ahead toe voor backward compatibility
        if 'grid_exchange_day_ahead' not in final_df.columns:
            final_df['grid_exchange_day_ahead'] = final_df.get('day_ahead_result', 0)
        
        # Dummy kolommen zijn al toegevoegd boven in de juiste volgorde
        
    else:
        # Fallback naar heuristiek
        if progress_callback:
            progress_callback("Pyomo optimalisatie gefaald. Gebruik fallback heuristiek...")
        final_df, total_cycles, network_violations = run_heuristic_fallback(df, config, progress_callback)
    
    # Progress update
    if progress_callback:
        progress_callback(f"Optimalisatie voltooid! Totaal cycli: {total_cycles:.2f}")
    
    # Bereken summary
    summary = {
        "total_revenue": 0,  # Niet relevant voor self-consumption
        "total_cost": 0,     # Niet relevant voor self-consumption
        "total_cycles": total_cycles,
        "cycle_history": [total_cycles],  # Eén waarde voor hele jaar
        "battery_power_MW": power_mw,
        "revenue_per_MW": 0,
        "network_violations": len(network_violations),
        "optimization_method": "Jaarlijkse lineaire optimalisatie",
        "warning_message": None
    }
    
    return final_df, summary



def run_heuristic_fallback(df, config, progress_callback=None):
    """Fallback heuristische methode als Pyomo faalt"""
    # Configuratie
    power_mw = config.POWER_MW
    capacity_mwh = config.CAPACITY_MWH
    eff_ch = config.EFF_CH
    eff_dis = config.EFF_DIS
    min_soc_frac = config.MIN_SOC
    max_soc_frac = config.MAX_SOC
    min_soc = min_soc_frac * capacity_mwh
    max_soc = max_soc_frac * capacity_mwh
    time_step_h = config.TIME_STEP_H
    max_cycles = config.MAX_CYCLES
    if hasattr(config, 'INIT_SOC'):
        current_soc = float(config.INIT_SOC) * capacity_mwh
    else:
        current_soc = 0.5 * capacity_mwh
    usable_capacity = capacity_mwh * (max_soc_frac - min_soc_frac)
    
    # Energiebelasting en leveringskosten (hergebruik van hoofdfunctie)
    tax_table = get_energy_tax_table()
    
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
    
    if len(df) > 0:
        avg_load_kwh = df['load'].mean()
        hours_per_year = 8760
        estimated_annual_consumption_mwh = (avg_load_kwh * hours_per_year) / 1000
        marginal_tax_rate = calculate_energy_tax(estimated_annual_consumption_mwh, tax_table)
    else:
        marginal_tax_rate = tax_table['consumption_brackets'][0]['tax_eur_per_mwh']

    # Initialisatie
    network_violations = []
    cycle_history = []
    cumulative_cycles = 0
    results = []

    # Verwerk elke dag
    total_days = len(df.groupby(pd.Grouper(freq='D')))
    day_count = 0
    
    for day, day_data in df.groupby(pd.Grouper(freq='D')):
        if len(day_data) == 0:
            continue
        
        day_count += 1
        # Toon alleen elke 10e dag of de laatste dag
        if day_count % 10 == 0 or day_count == total_days:
            msg = f"Fallback heuristiek: {day_count}/{total_days} dagen verwerkt... Cycli gebruikt: {cumulative_cycles:.1f}/{max_cycles}"
            if progress_callback:
                progress_callback(msg)
            else:
                print(msg, end='\r')

        soc_list = []
        charge_list = []
        discharge_list = []
        energy_charged_list = []
        energy_discharged_list = []
        soc_frac_list = []
        
        for i, (idx, row) in enumerate(day_data.iterrows()):
            grid_excl_battery = row['grid_excl_battery']  # kWh
            max_feed_in = row['max_feed_in_grid']  # kWh (negatief)
            max_take_from = row['max_take_from_grid']  # kWh (positief)
            
            charge_possible = 0
            discharge_possible = 0
            
            # Bepaal welke actie nodig is (laden OF ontladen, niet beide)
            action_needed = "none"
            required_charge_mw = 0
            required_discharge_mw = 0
            
            # Stap 1: Als er netlevering van PV is (grid_excl_battery negatief), laad om de PV op te slaan
            if grid_excl_battery < 0:
                # Bereken hoeveel we kunnen laden om de PV op te slaan
                available_pv_kwh = abs(grid_excl_battery)  # kWh
                required_charge_kwh = available_pv_kwh
                required_charge_mw = required_charge_kwh / time_step_h / 1000  # MW
                action_needed = "charge"
            
            # Stap 2: Los vermogensoverschrijdingen op (alleen als we nog geen actie hebben)
            if action_needed == "none":
                # Overschrijding afname: grid_excl_battery > max_take_from_grid
                if grid_excl_battery > max_take_from:
                    # We moeten ontladen om afname te verminderen
                    overschrijding_kwh = grid_excl_battery - max_take_from
                    required_discharge_kwh = overschrijding_kwh
                    required_discharge_mw = required_discharge_kwh / time_step_h / 1000  # MW
                    action_needed = "discharge"
                
                # Overschrijding invoeding: grid_excl_battery < max_feed_in_grid
                elif grid_excl_battery < max_feed_in:
                    # We moeten laden om invoeding te verminderen
                    overschrijding_kwh = abs(grid_excl_battery - max_feed_in)
                    required_charge_kwh = overschrijding_kwh
                    required_charge_mw = required_charge_kwh / time_step_h / 1000  # MW
                    action_needed = "charge"
            
            # Stap 3: Proactieve acties om toekomstige overschrijdingen te voorkomen
            if action_needed == "none":
                # Als we dicht bij de grenzen zitten, handel proactief
                margin_kwh = 10  # 10 kWh marge
                
                # Als we dicht bij afname overschrijding zitten
                if grid_excl_battery > (max_take_from - margin_kwh):
                    # Laad proactief om ruimte te maken
                    required_charge_kwh = margin_kwh
                    required_charge_mw = required_charge_kwh / time_step_h / 1000  # MW
                    action_needed = "charge"
                
                # Als we dicht bij invoeding overschrijding zitten
                elif grid_excl_battery < (max_feed_in + margin_kwh):
                    # Ontlaad proactief om ruimte te maken
                    required_discharge_kwh = margin_kwh
                    required_discharge_mw = required_discharge_kwh / time_step_h / 1000  # MW
                    action_needed = "discharge"
            
            # Voer de gekozen actie uit
            if action_needed == "charge":
                # Beperk door batterij vermogen en SoC
                max_charge_mw = min(power_mw, (max_soc - current_soc) / (time_step_h * eff_ch))
                charge_possible = min(required_charge_mw, max_charge_mw)
                discharge_possible = 0
            elif action_needed == "discharge":
                # Beperk door batterij vermogen en SoC
                max_discharge_mw = min(power_mw, (current_soc - min_soc) * eff_dis / time_step_h)
                discharge_possible = min(required_discharge_mw, max_discharge_mw)
                charge_possible = 0
            else:
                # Geen actie nodig
                charge_possible = 0
                discharge_possible = 0
            
            # Update SoC
            current_soc = current_soc + charge_possible * time_step_h * eff_ch - discharge_possible * time_step_h / eff_dis
            current_soc = min(max(current_soc, min_soc), max_soc)
            
            soc_list.append(current_soc)
            charge_list.append(charge_possible)
            discharge_list.append(discharge_possible)
            energy_charged_list.append(charge_possible * time_step_h * 1000)  # kWh
            energy_discharged_list.append(discharge_possible * time_step_h * 1000)  # kWh
            soc_frac_list.append((current_soc - min_soc) / (max_soc - min_soc))
        
        # Bereken cycli voor deze dag
        total_charged_energy = sum(energy_charged_list) / 1000  # MWh
        daily_cycle = (total_charged_energy * eff_ch) / usable_capacity if usable_capacity > 0 else 0
        cumulative_cycles += daily_cycle
        cycle_history.append(daily_cycle)
        
        day_data = day_data.copy()
        day_data['energy_charged_kWh'] = energy_charged_list
        day_data['energy_discharged_kWh'] = energy_discharged_list
        day_data['SoC_kWh'] = [s * 1000 for s in soc_list]
        day_data['SoC_pct'] = soc_frac_list
        
        # Controleer op netwerkoverschrijdingen na batterij acties
        for i, (idx, row) in enumerate(day_data.iterrows()):
            grid_excl_battery = row['grid_excl_battery']  # kWh
            max_feed_in = row['max_feed_in_grid']  # kWh
            max_take_from = row['max_take_from_grid']  # kWh
            
            # Netto uitwisseling met batterij
            net_battery_exchange = (energy_charged_list[i] - energy_discharged_list[i])  # kWh
            grid_incl_battery = grid_excl_battery + net_battery_exchange
            
            # Check overschrijdingen
            if max_feed_in < 0 and grid_incl_battery < max_feed_in:
                violation = {
                    'datetime': idx,
                    'type': 'feed_in',
                    'max_allowed': abs(max_feed_in),
                    'actual': abs(grid_incl_battery),
                    'violation': abs(grid_incl_battery) - abs(max_feed_in)
                }
                network_violations.append(violation)
            
            if max_take_from > 0 and grid_incl_battery > max_take_from:
                violation = {
                    'datetime': idx,
                    'type': 'take_from',
                    'max_allowed': max_take_from,
                    'actual': grid_incl_battery,
                    'violation': grid_incl_battery - max_take_from
                }
                network_violations.append(violation)
        
        results.append(day_data)

    final_df = pd.concat(results)
    total_cycles = cumulative_cycles
    
    # Voeg ontbrekende kolommen toe zoals in de hoofdfunctie
    # Voeg grid_exchange kolom toe (vervangt dummy) - toont werkelijke afname/invoeding van net na batterij
    if 'grid_exchange_kWh' not in final_df.columns:
        # Bereken net uitwisseling met grid: positief = afname van net, negatief = invoeding naar net
        final_df['grid_exchange_kWh'] = (final_df['load'] 
                                         - final_df['production_PV'] 
                                         + final_df['energy_charged_kWh']      # Laden = extra afname
                                         - final_df['energy_discharged_kWh'])  # Ontladen = minder afname
    
    # Voeg space kolommen toe indien nog niet aanwezig
    if 'space_for_charging_kWh' not in final_df.columns and 'space available for charging (kWh)' in final_df.columns:
        final_df['space_for_charging_kWh'] = final_df['space available for charging (kWh)']
    if 'space_for_discharging_kWh' not in final_df.columns and 'space available for discharging (kWh)' in final_df.columns:
        final_df['space_for_discharging_kWh'] = final_df['space available for discharging (kWh)']
    
    # Voeg uitgebreide kostenberekening toe (identiek aan hoofdfunctie)
    if 'total_energy_cost_self_consumption' not in final_df.columns:
        # Bereken totale energiekosten inclusief belasting en netkosten
        if 'price_day_ahead' in final_df.columns:
            # Netto grid verbruik = load - PV productie + batterij laden - batterij ontladen
            net_grid_consumption_kwh = (final_df['load'] 
                                       - final_df['production_PV'] 
                                       + final_df['energy_charged_kWh']      # Laden = extra inkoop (kWh)
                                       - final_df['energy_discharged_kWh'])  # Ontladen = minder inkoop of verkoop (kWh)
            
            net_grid_consumption_mwh = net_grid_consumption_kwh / 1000  # Converteer naar MWh
            
            # Bereken kosten per component
            day_ahead_costs = net_grid_consumption_mwh * final_df['price_day_ahead']
            
            # Voor positief verbruik (afname): voeg belasting, leveringskosten en transportkosten toe
            # Voor negatief verbruik (invoeding): alleen leveringskosten (energiebelasting en transportkosten niet van toepassing)
            tax_costs = np.where(net_grid_consumption_mwh > 0, net_grid_consumption_mwh * marginal_tax_rate, 0)
            
            # Transportkosten alleen voor netto afname van het net
            transport_costs_all = np.where(net_grid_consumption_mwh > 0, net_grid_consumption_mwh * transport_costs, 0)
            
            # Leveringskosten gelden voor absolute waarde van afname EN invoeding
            supply_costs_all = np.abs(net_grid_consumption_mwh) * supply_costs
            
            additional_costs = tax_costs + transport_costs_all + supply_costs_all
            
            # Voeg detail kolommen toe voor transparantie - EERST de kolommen aanmaken
            final_df['dummy1'] = 0
            final_df['dummy2'] = 0
            final_df['day_ahead_result'] = day_ahead_costs  # Kan positief (inkomsten) of negatief (kosten) zijn
            final_df['dummy3'] = 0
            final_df['energy_tax'] = np.where(net_grid_consumption_mwh > 0, -net_grid_consumption_mwh * marginal_tax_rate, 0)  # Altijd negatief (kosten)
            final_df['supplier_costs'] = -np.abs(net_grid_consumption_mwh) * supply_costs  # Altijd negatief (kosten)
            final_df['transport_costs'] = np.where(net_grid_consumption_mwh > 0, -net_grid_consumption_mwh * transport_costs, 0)  # Altijd negatief (kosten)
            
            # Totale kosten = day-ahead + alle extra kosten (nu met correcte voortekens)
            final_df['total_result_self_consumption'] = (day_ahead_costs + 
                                                         final_df['energy_tax'] + 
                                                         final_df['supplier_costs'] + 
                                                         final_df['transport_costs'])
        else:
            final_df['total_result_self_consumption'] = 0
            final_df['dummy1'] = 0
            final_df['dummy2'] = 0
            final_df['day_ahead_result'] = 0
            final_df['dummy3'] = 0
            final_df['energy_tax'] = 0
            final_df['supplier_costs'] = 0
            final_df['transport_costs'] = 0
    
    # Voeg grid_exchange_day_ahead toe voor backward compatibility
    if 'grid_exchange_day_ahead' not in final_df.columns:
        final_df['grid_exchange_day_ahead'] = final_df.get('day_ahead_result', 0)
    
    # Voeg dummy kolommen toe voor backward compatibility met bestaande code
    if 'dummy1' not in final_df.columns:
        final_df['dummy1'] = 0
    if 'dummy2' not in final_df.columns:
        final_df['dummy2'] = 0
    if 'dummy3' not in final_df.columns:
        final_df['dummy3'] = 0
    
    # Voeg nieuwe netwerk kolommen toe indien niet aanwezig
    if 'grid_excl_battery' not in final_df.columns:
        final_df['grid_excl_battery'] = 0
    if 'max_feed_in_grid' not in final_df.columns:
        final_df['max_feed_in_grid'] = 0
    if 'max_take_from_grid' not in final_df.columns:
        final_df['max_take_from_grid'] = 0
    
    return final_df, total_cycles, []

