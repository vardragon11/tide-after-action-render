# pages/scenario_builder.py

import streamlit as st
import json
import re
import subprocess

from Scenario2d import ScenarioScene
from Scenario3D import SpaceScenario3DScene
from pydantic import BaseModel
from collections import defaultdict
from typing import List, Optional

from scenario_utils import (
    Scenario,
    parse_scenario
)
# --- Page UI ---
st.title("🛰️ Scenario Builder")

#------Instructions
st.markdown("""
## User Script Instructions
Here's a set of guidelines for dictating or typing scenarios:

### General Principles
Structure: Organize your scenario using the keywords: Title, Description, Unit, Feature, Objective, Event.
Keywords: Start each object definition with its keyword followed by "is". For example, "Unit is ID equals U1..." or "Event is Time equals..."
Delimiter: Separate key-value pairs with commas (,).
Value Assignment: Assign values using "equals" or "=".

Consistency: Use consistent terminology. For instance, always refer to "Strength" and not "Power" when defining a unit.
Specific Object Instructions

### Unit
1. ID - Unique ID of Unit
2. Name - Unit Name 
3. Type -  Infantry, tanks, artillery, aircraft, naval ships, mechs, etc. 
4. Strength - The health of the unit 1-100 
5. Allegiance - It the unit friend or enmeny 
6. X - Based on a grid system position of unit on X plane
7. Y - Based on a grid system position of unit on X plane
8. Status - Current status of unit Active.
            
**Required: ID, Name, Type, Strength, Allegiance, X, Y.**
Optional: Status (defaults to "active").

### Feature:       
1. Feature - The type  of feature, Mountains, Building, Bunker, Beach.
2. X  - Based on a grid system position of unit on X plane
3. Y  - Based on a grid system position of unit on X plane
4. Size - scale based on map resolution (e.g., 10x10 grid).
            
**Required: Type, X(defaults 10.0), Y(defaults 10.0), Size.**

### Objective:
1. ID - Unique Identifier for Obj (Objective is ID equals O1.)
2. Desc equals evacuate to Boats.
3. X  - Based on a grid system position of unit on X plane
4. Y  - Based on a grid system position of unit on X plane
5. Priority - What level of important doees this objective carry (1-10)
            
**Required: ID, Desc (description).**
Optional: X, Y, Priority (defaults to 1).

### Event:
1. Event is Time equals 0.00.
2. Desc - description of the event "British Infantry fallback." 
3. Units - Who involved in the event
4. Type - The type of event (fire,move,hold)
5. X  - Based on a grid system position of unit on X plane
6. Y  - Based on a grid system position of unit on X plane
            
**Required: Time, Desc (description), Units, Type, X (defaults 0.0) , Y (defaults 0.0).**

### Example Scenario Script(Try me, script):  
            
`Title is Operation Coastal Shield. Description is Allied Forces are retreating under fire.
Unit is ID equals U1. Name equals British Infantry. Type equals Infantry. Strength equals 85. Allegiance equals Friendly. X equals 3. Y equals 6. Status equals Active.
Unit is ID equals U2. Name equals German Armor. Type equals Armor. Strength equals 92. Allegiance equals Enemy. X equals 3. Y equals 4. Status equals Active.
Unit is ID equals U3. Name equals French Infantry. Type equals Infantry. Strength equals 56. Allegiance equals Home. X equals 6. Y equals 4. Status equals Active.
Feature is Type equals Beach. X equals 5. Y equals 9.3. Size equals 50.
Objective is ID equals O1. Desc equals evacuate to Boats. X equals 5. Y equals 10. Priority equals 1.
Event is Time equals 0.00. Desc equals British Infantry fallback. Units equals U1. Type equals Move. X equals 5. Y equals 10.
Event is Time equals 0.01. Desc equals German Armor fires. Units equals U3. Type equals Fire. X equals 3. Y equals 7.
Event is Time equals 0.01. Desc equals French Infantry Fires. Units equals U2. Type equals Fire. X equals 5. Y equals 6.itle is Operation Coastal Shield.
Description is Allied forces retreat.`

### Reasoning:
Controlled Natural Language: These instructions define a controlled natural language (CNL) which balances human readability with machine parsability.
By following these instructions, users can create scenario script
s that the parser can interpret reliably, leading to more accurate and useful data extraction to dynamically create a simulation.""")

text_input = st.text_area("Enter scenario script:", height=300)

if st.button("🧠 Parse to JSON"):
    if text_input.strip():
        parsed = parse_scenario(text_input)
        st.success("Scenario parsed successfully!")
        st.json(parsed)
        with open("scenario.json", "w") as f:
            json.dump(parsed.dict(), f, indent=2)
        st.write("✅ Saved to `scenario.json`")
    else:
        st.warning("Please enter some scenario text.")

scene_type = st.selectbox("Choose scene type", ["2D Ground Scenario", "3D Space Scenario"])
if scene_type == "2D Ground Scenario":
    manim_class = "ScenarioScene"
    manim_file = "Scenario2d.py"
else:
    manim_class = "SpaceScenario3DScene"
    manim_file = "Scenario3D.py"

if st.button("🎬 Run Manim Animation"):
    result = subprocess.run(
        ["manim", "-pql", manim_file, manim_class],
        capture_output=True, text=True
    )
    st.text_area("📋 Manim Output Log", value=result.stdout + result.stderr, height=300)

    # Display rendered video
    video_path = f"media/videos/{manim_file.replace('.py','')}/480p15/{manim_class}.mp4"
    try:
        st.video(video_path)
        st.success("🎥 Playback complete.")
    except Exception as e:
        st.error(f"Could not load video: {e}")
