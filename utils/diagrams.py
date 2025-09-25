import base64
import os
from dash import html

def get_image_as_base64(path_in_assets):
    """
    Encodes an image from the assets folder to base64 for embedding in the SVG.
    """
    # In Dash, the app's root is the project folder, so we build the path from there.
    path = os.path.join('assets', path_in_assets)
    if not os.path.exists(path):
        print(f"Warning: Icon file not found at: {path}")
        return None
    with open(path, "rb") as f:
        data = f.read()
    return f"data:image/png;base64,{base64.b64encode(data).decode()}"

def create_horizontal_diagram_with_icons(situation_name):
    """
    Generates the correct horizontal diagram using PNG icons for any of the 7 situations.
    """
    # Note: Ensure these filenames match the files in your /assets/ folder.
    icons_b64 = {
        'grid': get_image_as_base64('power-line.png'),
        'meter': get_image_as_base64('energy-meter.png'),
        'alloc': get_image_as_base64('energy-meter.png'), # Using same icon for PAP/SAP
        'pv': get_image_as_base64('renewable-energy.png'),
        'batt': get_image_as_base64('energy-storage.png'),
        'load': get_image_as_base64('energy-consumption.png')
    }

    # Check if any icon failed to load, which can happen if files are missing.
    if any(v is None for v in icons_b64.values()):
        return html.Div(
            "Error: One or more icon files are missing from the 'assets' folder.",
            style={'color': 'red', 'fontWeight': 'bold', 'textAlign': 'center'}
        )

    # The rest of this function is copied directly from your Streamlit app's logic.
    arrow_defs = """
        <defs>
            <marker id="arrow-end-yellow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
                <path d="M 0 0 L 8 4 L 0 8 z" fill="#FDB813" />
            </marker>
            <marker id="arrow-start-yellow" viewBox="0 0 8 8" refX="1" refY="4" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
                <path d="M 8 0 L 0 4 L 8 8 z" fill="#FDB813" />
            </marker>
        </defs>
    """

    def create_node(x, y, label, icon_b64, w=100, h=80):
        return f'''
            <g transform="translate({x}, {y})">
                <rect x="0" y="0" width="{w}" height="{h}" rx="8" fill="#f8f9fa" stroke="#dee2e6" stroke-width="1"/>
                <image href="{icon_b64}" x="{w*0.25}" y="5" width="{w*0.5}" height="{h*0.5}"/>
                <text x="{w/2}" y="{h*0.8}" text-anchor="middle" font-weight="bold" font-size="12px" fill="#333">{label}</text>
            </g>
        '''
    
    POS = {
        'grid': (20, 185), 'main_meter': (180, 185), 'pv': (680, 20),
        'load': (680, 185), 'battery': (680, 350), 'meter_pv': (520, 20),
        'meter_battery': (520, 350), 'pap_main': (350, 185), 'sap1': (350, 80),
        'pap_center_sit6': (350, 185), 'sap2': (350, 290)
    }

    arrow = 'stroke="#FDB813" stroke-width="3" fill="none" marker-end="url(#arrow-end-yellow)"'
    arrow_two_way = 'stroke="#FDB813" stroke-width="3" fill="none" marker-start="url(#arrow-start-yellow)" marker-end="url(#arrow-end-yellow)"'
    direct_use_arrow = 'stroke="#FDB813" stroke-width="3" stroke-dasharray="6, 6" fill="none" marker-end="url(#arrow-end-yellow)"'

    nodes_to_draw = []
    lines_to_draw = []

    nodes_to_draw.extend([
        create_node(POS['grid'][0], POS['grid'][1], 'Grid', icons_b64['grid']),
        create_node(POS['main_meter'][0], POS['main_meter'][1], 'Main Meter', icons_b64['meter'])
    ])
    lines_to_draw.append(f'<line x1="{POS["grid"][0]+100}" y1="{POS["grid"][1]+40}" x2="{POS["main_meter"][0]}" y2="{POS["main_meter"][1]+40}" {arrow_two_way} />')
    
    svg_height = 450
    if "Situation 6" in situation_name:
        svg_height = 500

    if "Situation 1" in situation_name:
        nodes_to_draw.extend([ create_node(POS['pap_main'][0], POS['pap_main'][1], 'PAP', icons_b64['alloc']), create_node(POS['pv'][0], POS['pv'][1], 'PV', icons_b64['pv']), create_node(POS['load'][0], POS['load'][1], 'Load', icons_b64['load']), create_node(POS['meter_pv'][0], POS['meter_pv'][1], 'PV Meter', icons_b64['meter']) ])
        lines_to_draw.extend([ f'<line x1="{POS["main_meter"][0]+100}" y1="{POS["main_meter"][1]+40}" x2="{POS["pap_main"][0]}" y2="{POS["pap_main"][1]+40}" {arrow_two_way} />', f'<line x1="{POS["pap_main"][0]+100}" y1="{POS["pap_main"][1]+40}" x2="{POS["load"][0]}" y2="{POS["load"][1]+40}" {arrow} />', f'<path d="M {POS["meter_pv"][0]+45} {POS["meter_pv"][1]+80} L {POS["meter_pv"][0]+45} 145 L 400 145 L {POS["pap_main"][0]+50} {POS["pap_main"][1]}" {arrow} />', f'<line x1="{POS["pv"][0]}" y1="{POS["pv"][1]+40}" x2="{POS["meter_pv"][0]+100}" y2="{POS["meter_pv"][1]+40}" {arrow} />', f'<path d="M {POS["meter_pv"][0]+55} {POS["meter_pv"][1]+80} L {POS["meter_pv"][0]+55} 145 L {POS["load"][0]+50} 145 L {POS["load"][0]+50} {POS["load"][1]}" {direct_use_arrow} />' ])
    elif "Situation 2" in situation_name:
        nodes_to_draw.extend([ create_node(350, 80, 'SAP', icons_b64['alloc']), create_node(POS['meter_pv'][0], 80, 'PV Meter', icons_b64['meter']), create_node(POS['pv'][0], 80, 'PV', icons_b64['pv']), create_node(350, 290, 'PAP', icons_b64['alloc']), create_node(POS['load'][0], 290, 'Load', icons_b64['load']) ])
        lines_to_draw.extend([ f'<path d="M 350 120 L 315 120 L 315 200 L {POS["main_meter"][0]+100} 200" {arrow} />', f'<path d="M {POS["main_meter"][0]+100} 250 L 315 250 L 315 330 L 350 330" {arrow} />', f'<line x2="450" y1="120" x1="{POS["meter_pv"][0]}" y2="120" {arrow} />', f'<line x1="450" y1="330" x2="{POS["load"][0]}" y2="330" {arrow} />', f'<line x1="{POS["pv"][0]}" y1="120" x2="{POS["meter_pv"][0]+100}" y2="120" {arrow} />' ])
    elif "Situation 3" in situation_name:
        nodes_to_draw.extend([ create_node(350, 185, 'PAP', icons_b64['alloc']), create_node(350, 350, 'SAP', icons_b64['alloc']), create_node(POS['pv'][0], POS['pv'][1], 'PV', icons_b64['pv']), create_node(POS['load'][0], POS['load'][1], 'Load', icons_b64['load']), create_node(POS['battery'][0], POS['battery'][1], 'Battery', icons_b64['batt']), create_node(POS['meter_pv'][0], POS['meter_pv'][1], 'PV Meter', icons_b64['meter']), create_node(POS['meter_battery'][0], POS['meter_battery'][1], 'Battery Meter', icons_b64['meter']) ])
        lines_to_draw.extend([ f'<line x1="{POS["pv"][0]}" y1="{POS["pv"][1]+40}" x2="{POS["meter_pv"][0]+100}" y2="{POS["meter_pv"][1]+40}" {arrow} />', f'<path d="M {POS["meter_pv"][0]+45} {POS["meter_pv"][1]+80} L 565 145 L {POS["pap_main"][0]+50} 145 L {POS["pap_main"][0]+50} {POS["load"][1]}" {arrow} />', f'<line x1="{POS["main_meter"][0]+100}" y1="{POS["main_meter"][1]+40}" x2="350" y2="225" {arrow} />', f'<line x1="350" y1="200" x2="{POS["main_meter"][0]+100}" y2="200" {arrow} />', f'<line x1="450" y1="225" x2="{POS["load"][0]}" y2="{POS["load"][1]+40}" {arrow} />', f'<path d="M {POS["main_meter"][0]+100} 250 L 315 250 L 315 390 L 350 390" {arrow_two_way} />', f'<line x1="450" y1="390" x2="{POS["meter_battery"][0]}" y2="{POS["meter_battery"][1]+40}" {arrow_two_way} />', f'<line x1="{POS["meter_battery"][0]+100}" y1="{POS["meter_battery"][1]+40}" x2="{POS["battery"][0]}" y2="{POS["battery"][1]+40}" {arrow_two_way} />', f'<path d="M {POS["meter_pv"][0]+55} {POS["meter_pv"][1]+80} L 575 145 L {POS["load"][0]+50} 145 L {POS["load"][0]+50} {POS["load"][1]}" {direct_use_arrow} />' ])
    elif "Situation 4" in situation_name:
        nodes_to_draw.extend([ create_node(POS['pap_main'][0], POS['pap_main'][1], 'PAP', icons_b64['alloc']), create_node(POS['pv'][0], POS['pv'][1], 'PV', icons_b64['pv']), create_node(POS['load'][0], POS['load'][1], 'Load', icons_b64['load']), create_node(POS['battery'][0], POS['battery'][1], 'Battery', icons_b64['batt']), create_node(POS['meter_pv'][0], POS['meter_pv'][1], 'PV Meter', icons_b64['meter']), create_node(POS['meter_battery'][0], POS['meter_battery'][1], 'Battery Meter', icons_b64['meter']) ])
        lines_to_draw.extend([ f'<line x1="{POS["main_meter"][0]+100}" y1="{POS["main_meter"][1]+40}" x2="{POS["pap_main"][0]}" y2="{POS["pap_main"][1]+40}" {arrow} />', f'<path d="M {POS["meter_pv"][0]} 60 L 480 60 L 480 {POS["pap_main"][1]+20} L {POS["pap_main"][0]+100} {POS["pap_main"][1]+20}" {arrow} />', f'<line x1="{POS["pap_main"][0]+100}" y1="{POS["pap_main"][1]+40}" x2="{POS["load"][0]}" y2="{POS["load"][1]+40}" {arrow} />', f'<path d="M {POS["pap_main"][0]+100} {POS["pap_main"][1]+60} L 480 {POS["pap_main"][1]+60} L 480 390 L {POS["meter_battery"][0]} 390" {arrow} />', f'<line x1="{POS["pv"][0]}" y1="{POS["pv"][1]+40}" x2="{POS["meter_pv"][0]+100}" y2="{POS["meter_pv"][1]+40}" {arrow} />', f'<line x1="{POS["meter_battery"][0]+100}" y1="{POS["meter_battery"][1]+40}" x2="{POS["battery"][0]}" y2="{POS["battery"][1]+40}" {arrow_two_way} />', f'<path d="M {POS["meter_pv"][0]+50} {POS["meter_pv"][1]+80} L {POS["meter_pv"][0]+50} 200 L {POS["load"][0]} 200" {direct_use_arrow} />', f'<path d="M {POS["meter_battery"][0]+50} {POS["meter_battery"][1]} L {POS["meter_battery"][0]+50} 250 L {POS["load"][0]} 250" {direct_use_arrow} />', f'<path d="M {POS["meter_pv"][0]} {POS["meter_pv"][1]+20} L {POS["meter_pv"][0]-50} {POS["meter_pv"][1]+20} L {POS["meter_pv"][0]-50} {POS["meter_battery"][1]+60} L {POS["meter_battery"][0]} {POS["meter_battery"][1]+60}" {direct_use_arrow} />' ])
    elif "Situation 5" in situation_name:
        nodes_to_draw.extend([ create_node(POS['pv'][0], POS['pv'][1], 'PV', icons_b64['pv']), create_node(POS['meter_pv'][0], POS['meter_pv'][1], 'PV Meter', icons_b64['meter']), create_node(350, 185, 'SAP', icons_b64['alloc']), create_node(POS['meter_battery'][0], 185, 'Battery Meter', icons_b64['meter']), create_node(POS['battery'][0], 185, 'Battery', icons_b64['batt']), create_node(350, 350, 'PAP', icons_b64['alloc']), create_node(POS['load'][0], 350, 'Load', icons_b64['load']) ])
        lines_to_draw.extend([ f'<line x1="{POS["pv"][0]}" y1="{POS["pv"][1]+40}" x2="{POS["meter_pv"][0]+100}" y2="{POS["meter_pv"][1]+40}" {arrow} />', f'<line x1="{POS["meter_pv"][0]}" y1="{POS["meter_pv"][1]+40}" x2="450" y2="200" {arrow} />', f'<line x2="{POS["main_meter"][0]+100}" y1="200" x1="350" y2="200" {arrow} />', f'<line x2="350" y1="225" x1="{POS["main_meter"][0]+100}" y2="225" {arrow} />', f'<line x1="450" y1="225" x2="{POS["meter_battery"][0]}" y2="225" {arrow_two_way} />', f'<line x1="{POS["meter_battery"][0]+100}" y1="230" x2="{POS["battery"][0]}" y2="230" {arrow_two_way} />', f'<path d="M {POS["main_meter"][0]+100} 250 L 315 250 L 315 390 L 350 390" {arrow} />', f'<line x1="450" y1="390" x2="{POS["load"][0]}" y2="390" {arrow} />', f'<path d="M {POS["pv"][0]} {POS["pv"][1]+60} C 640 120, 640 200, {POS["battery"][0]} {POS["battery"][1]-145}" {direct_use_arrow} />' ])
    elif "Situation 6" in situation_name:
        nodes_to_draw.extend([ create_node(POS['sap1'][0], POS['sap1'][1], 'SAP1', icons_b64['alloc']), create_node(POS['pap_center_sit6'][0], POS['pap_center_sit6'][1], 'PAP', icons_b64['alloc']), create_node(POS['sap2'][0], POS['sap2'][1], 'SAP2', icons_b64['alloc']), create_node(POS['pv'][0], POS['pv'][1], 'PV', icons_b64['pv']), create_node(POS['load'][0], POS['load'][1], 'Load', icons_b64['load']), create_node(POS['battery'][0], POS['battery'][1], 'Battery', icons_b64['batt']), create_node(POS['meter_pv'][0], POS['meter_pv'][1], 'PV Meter', icons_b64['meter']), create_node(POS['meter_battery'][0], POS['meter_battery'][1], 'Battery Meter', icons_b64['meter']) ])
        lines_to_draw.extend([ f'<path d="M {POS["main_meter"][0]+100} {POS["main_meter"][1]+20} L 315 {POS["main_meter"][1]+20} L 315 {POS["sap1"][1]+40} L {POS["sap1"][0]} {POS["sap1"][1]+40}" {arrow} />', f'<line x1="{POS["main_meter"][0]+100}" y1="{POS["main_meter"][1]+40}" x2="{POS["pap_center_sit6"][0]}" y2="{POS["pap_center_sit6"][1]+40}" {arrow} />', f'<path d="M {POS["main_meter"][0]+100} {POS["main_meter"][1]+60} L 315 {POS["main_meter"][1]+60} L 315 {POS["sap2"][1]+40} L {POS["sap2"][0]} {POS["sap2"][1]+40}" {arrow} />', f'<line x1="{POS["meter_pv"][0]}" y1="{POS["meter_pv"][1]+40}" x2="{POS["sap1"][0]+100}" y2="{POS["sap1"][1]+40}" {arrow} />', f'<line x1="{POS["pap_center_sit6"][0]+100}" y1="{POS["pap_center_sit6"][1]+40}" x2="{POS["load"][0]}" y2="{POS["load"][1]+40}" {arrow} />', f'<line x1="{POS["sap2"][0]+100}" y1="{POS["sap2"][1]+40}" x2="{POS["meter_battery"][0]}" y2="{POS["meter_battery"][1]+40}" {arrow_two_way} />', f'<line x1="{POS["pv"][0]}" y1="{POS["pv"][1]+40}" x2="{POS["meter_pv"][0]+100}" y2="{POS["meter_pv"][1]+40}" {arrow} />', f'<line x1="{POS["meter_battery"][0]+100}" y1="{POS["meter_battery"][1]+40}" x2="{POS["battery"][0]}" y2="{POS["battery"][1]+40}" {arrow_two_way} />' ])
    elif "Situation 7" in situation_name:
        nodes_to_draw.extend([ create_node(POS['pap_main'][0], POS['pap_main'][1], 'PAP', icons_b64['alloc']), create_node(POS['pv'][0], POS['pv'][1], 'PV', icons_b64['pv']), create_node(POS['battery'][0], POS['battery'][1], 'Battery', icons_b64['batt']), create_node(POS['meter_pv'][0], POS['meter_pv'][1], 'PV Meter', icons_b64['meter']), create_node(POS['meter_battery'][0], POS['meter_battery'][1], 'Battery Meter', icons_b64['meter']) ])
        lines_to_draw.extend([ f'<line x1="{POS["main_meter"][0]+100}" y1="{POS["main_meter"][1]+40}" x2="{POS["pap_main"][0]}" y2="{POS["pap_main"][1]+40}" {arrow} />', f'<path d="M {POS["meter_pv"][0]} 60 L 480 60 L 480 {POS["pap_main"][1]+20} L {POS["pap_main"][0]+100} {POS["pap_main"][1]+20}" {arrow} />', f'<path d="M {POS["pap_main"][0]+100} {POS["pap_main"][1]+60} L 480 {POS["pap_main"][1]+60} L 480 390 L {POS["meter_battery"][0]} 390" {arrow} />', f'<line x2="{POS["meter_pv"][0]+100}" y1="{POS["meter_pv"][1]+40}" x1="{POS["pv"][0]}" y2="{POS["pv"][1]+40}" {arrow} />', f'<line x1="{POS["meter_battery"][0]+100}" y1="{POS["meter_battery"][1]+40}" x2="{POS["battery"][0]}" y2="{POS["battery"][1]+40}" {arrow} />', f'<path d="M {POS["meter_pv"][0]} {POS["meter_pv"][1]+20} L {POS["meter_pv"][0]-50} {POS["meter_pv"][1]+20} L {POS["meter_pv"][0]-50} {POS["battery"][1]+60} L {POS["meter_battery"][0]} {POS["battery"][1]+60}" {direct_use_arrow} />' ])

    svg_content = "".join(nodes_to_draw) + "".join(lines_to_draw)
    
    # We create the full SVG string and pass it to an Iframe's srcDoc attribute.
    # This is the standard way to render raw HTML/SVG in Dash.
    svg_html_string = f'''
        <div style="width: 100%; max-width: 850px; height: {svg_height}px; font-family: sans-serif; margin: auto; border: 1px solid #ddd; border-radius: 8px;">
            <svg viewBox="0 0 850 {svg_height}" style="width: 100%; height: 100%;">
                {arrow_defs}
                {svg_content}
            </svg>
        </div>
    '''
    
    return html.Iframe(
        srcDoc=svg_html_string,
        style={"width": "100%", "height": f"{svg_height + 20}px", "border": 0}
    )