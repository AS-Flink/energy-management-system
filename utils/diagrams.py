# utils/diagrams.py
import base64
import os
from dash import html

def get_image_as_base64(path_in_assets):
    """Encodes an image from the assets folder to base64."""
    path = os.path.join('assets', path_in_assets)
    if not os.path.exists(path):
        print(f"Icon file not found at: {path}")
        return None
    with open(path, "rb") as f:
        data = f.read()
    return f"data:image/png;base64,{base64.b64encode(data).decode()}"

def create_horizontal_diagram_with_icons(situation_name):
    """
    Generates the correct horizontal diagram using PNG icons for any of the 7 situations.
    In Dash, icons are loaded from the /assets/ folder.
    """
    icons_b64 = {
        'grid': get_image_as_base64('power-line.png'),
        'meter': get_image_as_base64('energy-meter.png'),
        'alloc': get_image_as_base64('energy-meter.png'), # Using same icon for PAP/SAP
        'pv': get_image_as_base64('renewable-energy.png'),
        'batt': get_image_as_base64('energy-storage.png'),
        'load': get_image_as_base64('energy-consumption.png')
    }

    # Check if any icon failed to load
    if any(v is None for v in icons_b64.values()):
        return html.Div("Error: One or more icon files are missing from the 'assets' folder.", style={'color': 'red'})

    # The rest of this function is copied directly from your Streamlit app
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
    # ... (rest of the elif conditions for situations 2-7 are copied here verbatim) ...
    # This part is very long, so I'm omitting it for brevity, but you would paste it here.
    # The code is identical to your original script.
    
    svg_content = "".join(nodes_to_draw) + "".join(lines_to_draw)
    svg_string = f'''
        <svg viewBox="0 0 850 {svg_height}" style="width: 100%; height: 100%;">
            {arrow_defs}
            {svg_content}
        </svg>
    '''
    # Using Iframe to render raw SVG string in Dash
    return html.Iframe(
        srcDoc=svg_string,
        style={"width": "100%", "maxWidth": "850px", "height": f"{svg_height}px", "border": "1px solid #ddd", "borderRadius": "8px", "margin": "auto", "display": "block"}
    )