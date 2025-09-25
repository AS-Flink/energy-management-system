import pandas as pd
import numpy as np
import numpy_financial as npf

HARDCODED_DEFAULTS = {
    # General & Financial
    'project_term': 10, 'lifespan_battery_tech': 10, 'lifespan_pv_tech': 25,
    'depr_period_battery': 10, 'depr_period_pv': 15, 'debt_senior_pct': 0.0,
    'debt_junior_pct': 0.0, 'irr_equity_req': 0.10, 'interest_rate_senior': 0.06,
    'interest_rate_junior': 0.08, 'term_senior': 10, 'term_junior': 10, 'wacc': 0.1,
    'eia_pct': 0.4,

    # Indexations
    'inflation': 0.02, 'idx_trading_income': -0.02, 'idx_supplier_costs': 0.0,
    'idx_om_bess': 0.0, 'idx_om_pv': 0.0, 'idx_grid_op': 0.005, 'idx_other_costs': 0.0,
    'idx_ppa_income': 0.0, 'idx_curtailment_income': 0.0,
    
    # Grid Connection
    'grid_one_time_bess': 0.0, 'grid_one_time_pv': 0.0, 'grid_one_time_general': 0.0,
    'grid_annual_fixed': 0.0, 'grid_annual_kw_max': 2000.0,
    'grid_annual_kw_contract': 800.0, 'grid_annual_kwh_offtake': 5000.0,

    # BESS
    'bess_power_kw': 2000, 'bess_capacity_kwh': 4000, 'bess_min_soc': 0.05,
    'bess_max_soc': 0.95, 'bess_charging_eff': 0.92, 'bess_discharging_eff': 0.92,
    'bess_annual_degradation': 0.04, 'bess_cycles_per_year': 600,
    'bess_capex_per_kwh': 116.3, 'bess_capex_civil_pct': 0.06, 'bess_capex_it_per_kwh': 1.5,
    'bess_capex_security_per_kwh': 5.0, 'bess_capex_permits_pct': 0.015,
    'bess_capex_mgmt_pct': 0.025, 'bess_capex_contingency_pct': 0.05,
    'bess_income_trading_per_mw_year': 243254, 'bess_income_ctrl_party_pct': 0.1,
    'bess_income_supplier_cost_per_mwh': 2.0, 'bess_opex_om_per_year': 4652.0,
    'bess_opex_asset_mgmt_per_mw_year': 4000.0, 'bess_opex_insurance_pct': 0.01,
    'bess_opex_property_tax_pct': 0.001, 'bess_opex_overhead_per_kwh_year': 1.0,
    'bess_opex_other_per_kwh_year': 1.0, 'bess_opex_retribution': 0.0,

    # PV
    'pv_power_per_panel_wp': 590, 'pv_panel_count': 3479, 'pv_full_load_hours': 817.8,
    'pv_annual_degradation': 0.005, 'pv_capex_per_wp': 0.2, 'pv_capex_civil_pct': 0.08,
    'pv_capex_security_pct': 0.02, 'pv_capex_permits_pct': 0.01,
    'pv_capex_mgmt_pct': 0.025, 'pv_capex_contingency_pct': 0.05,
    'pv_income_ppa_per_mwp': 0.0, 'pv_income_ppa_per_kwh': 0.0,
    'pv_income_curtailment_per_mwp': 0.0, 'pv_income_curtailment_per_kwh': 0.0,
    'pv_opex_insurance_pct': 0.01, 'pv_opex_property_tax_pct': 0.001,
    'pv_opex_overhead_pct': 0.005, 'pv_opex_other_pct': 0.005,
    'pv_opex_om_y1': 0.0, 'pv_opex_retribution': 0.0
}

def calculate_all_kpis(i, tech_type):
    kpis = {}
    if tech_type == 'bess':
        kpis['Capacity Factor'] = i['bess_capacity_kwh'] / i['bess_power_kw'] if i['bess_power_kw'] > 0 else 0
        kpis['SoC Available'] = i['bess_max_soc'] - i['bess_min_soc']
        kpis['Usable Capacity'] = i['bess_capacity_kwh'] * kpis['SoC Available']
        kpis['C-Rate'] = i['bess_power_kw'] / kpis['Usable Capacity'] if kpis['Usable Capacity'] > 0 else 0
        kpis['Round Trip Efficiency (RTE)'] = i['bess_charging_eff'] * i['bess_discharging_eff']
        kpis['Purchase Costs'] = i['bess_capacity_kwh'] * i['bess_capex_per_kwh']
        kpis['IT & Security Costs'] = i['bess_capacity_kwh'] * (i['bess_capex_it_per_kwh'] + i['bess_capex_security_per_kwh'])
        base_capex = kpis['Purchase Costs'] + kpis['IT & Security Costs']
        kpis['Civil Works'] = base_capex * i['bess_capex_civil_pct']
        capex_subtotal = base_capex + kpis['Civil Works']
        kpis['Permits & Fees'] = capex_subtotal * i['bess_capex_permits_pct']
        kpis['Project Management'] = capex_subtotal * i['bess_capex_mgmt_pct']
        kpis['Contingency'] = capex_subtotal * i['bess_capex_contingency_pct']
        kpis['total_capex'] = capex_subtotal + kpis['Permits & Fees'] + kpis['Project Management'] + kpis['Contingency']
        kpis['trading_income_y1'] = (i['bess_power_kw'] / 1000) * i['bess_income_trading_per_mw_year']
        kpis['control_party_costs_y1'] = kpis['trading_income_y1'] * i['bess_income_ctrl_party_pct']
        offtake_y1 = i['bess_cycles_per_year'] * kpis['Usable Capacity'] / i['bess_charging_eff']
        kpis['supplier_costs_y1'] = (offtake_y1 / 1000) * i['bess_income_supplier_cost_per_mwh']
        kpis['om_y1'] = i['bess_opex_om_per_year']
        kpis['retribution_y1'] = i['bess_opex_retribution']
        kpis['asset_mgmt_y1'] = (i['bess_power_kw'] / 1000) * i['bess_opex_asset_mgmt_per_mw_year']
        kpis['insurance_y1'] = kpis['total_capex'] * i['bess_opex_insurance_pct']
        kpis['property_tax_y1'] = kpis['total_capex'] * i['bess_opex_property_tax_pct']
        kpis['overhead_y1'] = i['bess_capacity_kwh'] * i['bess_opex_overhead_per_kwh_year']
        kpis['other_y1'] = i['bess_capacity_kwh'] * i['bess_opex_other_per_kwh_year']
    elif tech_type == 'pv':
        kpis['Total Peak Power'] = (i['pv_power_per_panel_wp'] * i['pv_panel_count']) / 1000
        kpis['Production (Year 1)'] = kpis['Total Peak Power'] * i['pv_full_load_hours']
        kpis['Purchase Costs'] = kpis['Total Peak Power'] * 1000 * i['pv_capex_per_wp']
        capex_subtotal = kpis['Purchase Costs']
        kpis['Civil Works'] = capex_subtotal * i['pv_capex_civil_pct']
        capex_subtotal += kpis['Civil Works']
        kpis['Security'] = capex_subtotal * i['pv_capex_security_pct']
        kpis['Permits & Fees'] = capex_subtotal * i['pv_capex_permits_pct']
        kpis['Project Management'] = capex_subtotal * i['pv_capex_mgmt_pct']
        kpis['Contingency'] = capex_subtotal * i['pv_capex_contingency_pct']
        kpis['total_capex'] = capex_subtotal + kpis['Security'] + kpis['Permits & Fees'] + kpis['Project Management'] + kpis['Contingency']
        kpis['ppa_income_y1'] = (kpis['Total Peak Power'] * 1000 * i['pv_income_ppa_per_mwp']) + (kpis['Production (Year 1)'] * i['pv_income_ppa_per_kwh'])
        kpis['curtailment_income_y1'] = (kpis['Total Peak Power'] * 1000 * i['pv_income_curtailment_per_mwp']) + (kpis['Production (Year 1)'] * i['pv_income_curtailment_per_kwh'])
        kpis['om_y1'] = i['pv_opex_om_y1']
        kpis['retribution_y1'] = i['pv_opex_retribution']
        kpis['insurance_y1'] = kpis['total_capex'] * i['pv_opex_insurance_pct']
        kpis['property_tax_y1'] = kpis['total_capex'] * i['pv_opex_property_tax_pct']
        kpis['overhead_y1'] = kpis['total_capex'] * i['pv_opex_overhead_pct']
        kpis['other_y1'] = kpis['total_capex'] * i['pv_opex_other_pct']
    return kpis

def run_financial_model(i, project_type):
    years_op = np.arange(1, int(i['project_term']) + 1)
    df = pd.DataFrame(index=years_op); df.index.name = 'Year'
    is_bess, is_pv = 'BESS' in project_type, 'PV' in project_type
    years_vector = df.index - 1
    
    df['idx_inflation'] = (1 + i['inflation']) ** years_vector
    df['idx_trading_income'] = (1 + i['inflation'] + i['idx_trading_income']) ** years_vector
    df['idx_supplier_costs'] = (1 + i['inflation'] + i['idx_supplier_costs']) ** years_vector
    df['idx_om_bess'] = (1 + i['inflation'] + i['idx_om_bess']) ** years_vector
    df['idx_om_pv'] = (1 + i['inflation'] + i['idx_om_pv']) ** years_vector
    df['idx_grid_op'] = (1 + i['inflation'] + i['idx_grid_op']) ** years_vector
    df['idx_other_costs'] = (1 + i['inflation'] + i['idx_other_costs']) ** years_vector
    df['idx_ppa_income'] = (1 + i['inflation'] + i['idx_ppa_income']) ** years_vector
    df['idx_curtailment_income'] = (1 + i['inflation'] + i['idx_curtailment_income']) ** years_vector
    df['idx_degradation_bess'] = (1 - i['bess_annual_degradation']) ** df.index
    df['idx_degradation_pv'] = (1 - i['pv_annual_degradation']) ** df.index

    bess_base = calculate_all_kpis(i, 'bess') if is_bess else {}
    bess_capex = bess_base.get('total_capex', 0)
    bess_active_mask = (df.index <= i['project_term']) & (df.index <= i['lifespan_battery_tech'])
    if is_bess:
        df['bess_trading_income'] = bess_base['trading_income_y1'] * df['idx_trading_income'] * df['idx_degradation_bess']
        df['bess_control_party_costs'] = -df['bess_trading_income'] * i['bess_income_ctrl_party_pct']
        df['bess_supplier_costs'] = -bess_base['supplier_costs_y1'] * df['idx_supplier_costs'] * df['idx_degradation_bess']
        df['bess_om'] = -bess_base['om_y1'] * df['idx_om_bess']
        df['bess_retribution'] = -bess_base['retribution_y1'] * df['idx_other_costs']
        df['bess_asset_mgmt'] = -bess_base['asset_mgmt_y1'] * df['idx_other_costs']
        df['bess_insurance'] = -bess_base['insurance_y1'] * df['idx_other_costs']
        df['bess_property_tax'] = -bess_base['property_tax_y1'] * df['idx_other_costs']
        df['bess_overhead'] = -bess_base['overhead_y1'] * df['idx_other_costs']
        df['bess_other'] = -bess_base['other_y1'] * df['idx_other_costs']
        bess_cols = [c for c in df.columns if 'bess_' in c and 'idx_' not in c]
        for col in bess_cols: df[col] *= bess_active_mask
        df['ebitda_bess'] = df[bess_cols].sum(axis=1)
    else: df['ebitda_bess'] = 0

    pv_base = calculate_all_kpis(i, 'pv') if is_pv else {}
    pv_capex = pv_base.get('total_capex', 0)
    pv_active_mask = (df.index <= i['project_term']) & (df.index <= i['lifespan_pv_tech'])
    if is_pv:
        df['pv_ppa_income'] = pv_base['ppa_income_y1'] * df['idx_ppa_income'] * df['idx_degradation_pv']
        df['pv_curtailment_income'] = pv_base['curtailment_income_y1'] * df['idx_curtailment_income'] * df['idx_degradation_pv']
        df['pv_om'] = -pv_base['om_y1'] * df['idx_om_pv']
        df['pv_retribution'] = -pv_base['retribution_y1'] * df['idx_other_costs']
        df['pv_insurance'] = -pv_base['insurance_y1'] * df['idx_other_costs']
        df['pv_property_tax'] = -pv_base['property_tax_y1'] * df['idx_other_costs']
        df['pv_overhead'] = -pv_base['overhead_y1'] * df['idx_other_costs']
        df['pv_other'] = -pv_base['other_y1'] * df['idx_other_costs']
        pv_cols = [c for c in df.columns if 'pv_' in c and 'idx_' not in c]
        for col in pv_cols: df[col] *= pv_active_mask
        df['ebitda_pv'] = df[pv_cols].sum(axis=1)
    else: df['ebitda_pv'] = 0

    grid_capex = i['grid_one_time_bess'] + i['grid_one_time_pv'] + i['grid_one_time_general']
    df['grid_annual_fixed'] = -i['grid_annual_fixed'] * df['idx_grid_op']
    df['grid_annual_kw_max'] = -i['grid_annual_kw_max'] * df['idx_grid_op']
    df['grid_annual_kw_contract'] = -i['grid_annual_kw_contract'] * df['idx_grid_op']
    df['grid_annual_kwh_offtake'] = -i['grid_annual_kwh_offtake'] * df['idx_grid_op']
    df['ebitda_grid'] = df[['grid_annual_fixed', 'grid_annual_kw_max', 'grid_annual_kw_contract', 'grid_annual_kwh_offtake']].sum(axis=1)
    
    df['total_ebitda'] = df['ebitda_bess'] + df['ebitda_pv'] + df['ebitda_grid']
    total_investment = bess_capex + pv_capex + grid_capex
    df['depreciation_bess'] = -bess_capex / i['depr_period_battery'] if i['depr_period_battery'] > 0 else 0
    df.loc[df.index > i['depr_period_battery'], 'depreciation_bess'] = 0
    df['depreciation_pv'] = -pv_capex / i['depr_period_pv'] if i['depr_period_pv'] > 0 else 0
    df.loc[df.index > i['depr_period_pv'], 'depreciation_pv'] = 0
    
    df['result_before_eia'] = df['total_ebitda'] + df['depreciation_bess'] + df['depreciation_pv']
    eia_allowance = total_investment * i['eia_pct']
    df['eia_applied'] = 0
    if len(df) > 0 and df.loc[1, 'result_before_eia'] > 0:
        df.loc[1, 'eia_applied'] = min(eia_allowance, df.loc[1, 'result_before_eia'])
        
    df['result_before_tax'] = df['result_before_eia'] - df['eia_applied']
    df['corporate_tax'] = -df['result_before_tax'].apply(lambda x: 200000 * 0.19 + (x - 200000) * 0.258 if x > 200000 else x * 0.19 if x > 0 else 0)
    df['profit_after_tax'] = df['result_before_tax'] + df['corporate_tax']
    
    df['net_cash_flow'] = df['total_ebitda'] + df['corporate_tax']
    ncf_y0 = -total_investment
    df['cumulative_cash_flow'] = df['net_cash_flow'].cumsum() + ncf_y0
    df['cumulative_ebitda'] = df['total_ebitda'].cumsum() + ncf_y0
    
    cash_flows_for_irr = [ncf_y0] + df['net_cash_flow'].tolist()
    
    metrics = {}
    metrics['total_investment'] = total_investment
    metrics['cumulative_cash_flow_end'] = df['cumulative_cash_flow'].iloc[-1] if not df.empty else ncf_y0
    metrics['cumulative_ebitda_end'] = df['cumulative_ebitda'].iloc[-1] if not df.empty else ncf_y0
    
    metrics['npv'] = npf.npv(i['wacc'], cash_flows_for_irr[1:]) + ncf_y0 if i['wacc'] > -1 else "Invalid WACC"
    
    try:
        metrics['equity_irr'] = npf.irr(cash_flows_for_irr)
    except Exception:
        metrics['equity_irr'] = "Cannot calculate"
        
    project_ebitda_flows = [ncf_y0] + df['total_ebitda'].tolist()
    try:
        metrics['project_irr'] = npf.irr(project_ebitda_flows)
    except Exception:
        metrics['project_irr'] = "Cannot calculate"
        
    try:
        payback_year_val = df[df['cumulative_cash_flow'] >= 0].index[0]
        cash_flow_prev_year = df.loc[payback_year_val - 1, 'cumulative_cash_flow'] if payback_year_val > 1 else ncf_y0
        cash_flow_payback_year = df.loc[payback_year_val, 'net_cash_flow']
        
        if cash_flow_payback_year == 0:
            metrics['payback_period'] = payback_year_val
        else:
            metrics['payback_period'] = (payback_year_val - 1) + abs(cash_flow_prev_year / cash_flow_payback_year)
    except (IndexError, KeyError, ZeroDivisionError):
        metrics['payback_period'] = "Not reached"
        
    return {"df": df, "metrics": metrics, "bess_kpis": bess_base, "pv_kpis": pv_base}