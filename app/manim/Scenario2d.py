import json
from manim import *

class ScenarioScene(Scene):
    def construct(self):
        # Load scenario data
        with open("scenario.json", "r") as f:
            data = json.load(f)

        self.unit_mobs = {}

        # Create the grid
        grid = NumberPlane(
            x_range=[-1, data["terrain"]["dimensions"][0] + 1, 1],
            y_range=[-1, data["terrain"]["dimensions"][1] + 2, 1],
            background_line_style={"stroke_opacity": 0.2}
        ).scale(0.6)
        self.add(grid)

        # Title
        title = Text(data["title"].strip("."), font_size=32).to_edge(UP)
        self.play(Write(title))

       
        # Terrain color mapping
        TERRAIN_COLORS = {
            "Beach": LIGHT_BROWN,
            "Forest": GREEN,
            "Mountain": GRAY,
            "River": BLUE,
            "Urban": DARK_GRAY,
            "Default": LIGHT_GRAY
        }
        # --- Terrain features ---
        for feature in data["terrain"]["features"]:
          t_type = feature.get("type", "Default")
          color = TERRAIN_COLORS.get(t_type, TERRAIN_COLORS["Default"])

          # You can customize shape based on type too, but using Rectangle for now
          terrain_shape = Rectangle(width=2, height=1, color=color, fill_opacity=0.3)
          terrain_shape.move_to(grid.c2p(*feature["position"]))

          label = Text(t_type, font_size=16).next_to(terrain_shape, UP, buff=0.1)
          self.play(FadeIn(terrain_shape), Write(label))

        # --- Objectives ---
        for obj in data["objectives"]:
            dot = Dot(grid.c2p(*obj["location"]), color=ORANGE)
            label = Text(obj["description"], font_size=16).next_to(dot, RIGHT, buff=0.1)
            group = VGroup(dot, label)
            # Add to scene
            self.play(FadeIn(group))
            # Wait for 2 seconds
            self.wait(2)                  
            self.play(FadeOut(group)) 

        # --- Units ---
        for unit in data["units"]:
            pos = unit["position"]
            color = BLUE if unit["allegiance"].lower() == "Friendly" else RED
            unit_circle = Circle(radius=0.25, color=color, fill_opacity=0.6)
            unit_circle.move_to(grid.c2p(*pos))
            label = Text(unit["id"], font_size=14).move_to(unit_circle.get_center())
            unit_group = VGroup(unit_circle, label)
            self.play(FadeIn(unit_group), run_time=0.3)
            self.unit_mobs[unit["id"]] = unit_group

        # --- Timeline Animation w/ explicit location ---
        self.wait(0.5)
        for event in data["timeline"]:
            desc = event["description"]
            display_text = Text(desc, font_size=20).to_edge(DOWN)
            self.play(Write(display_text), run_time=0.5)

            for uid in event["involved_units"]:
                mob = self.unit_mobs.get(uid)
                if not mob:
                    continue

                self.play(Indicate(mob), run_time=0.4)

                # Move if event has a location
                if event.get("location"):
                    dest_point = grid.c2p(*event["location"])
                    self.play(mob.animate.move_to(dest_point), run_time=1)

                # Optional: simple animation for "fire" event type
                if event.get("event_type") == "fire" and event.get("location"):
                  firing_pos = mob.get_center()
                  target_pos = grid.c2p(*event["location"])

                  # Create the line and fire dot
                  line = Line(firing_pos, target_pos, color=WHITE)
                  fire_dot = Dot(color=YELLOW).move_to(firing_pos)

                  self.add(line, fire_dot)
                  self.play(MoveAlongPath(fire_dot, line), run_time=0.6)
                  self.remove(fire_dot, line)

            self.remove(display_text)
            self.wait(0.3)

        self.wait(2)