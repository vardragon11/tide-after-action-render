import json
from manim import *

class SpaceScenario3DScene(ThreeDScene):
    def construct(self):
        # Load JSON data
        with open("scenario.json", "r") as f:
            data = json.load(f)

        self.unit_mobs = {}

        # Setup 3D camera
        self.set_camera_orientation(phi=70 * DEGREES, theta=45 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.02)

        # Draw Earth
        earth = Sphere(radius=1, color=BLUE, resolution=(24, 48))
        earth.set_fill(BLUE_E, opacity=0.5)
        earth.set_shade_in_3d(True)
        earth_label = Text("Earth", font_size=24).move_to([0, -1.4, 1.2]).scale(0.4)
        self.play(FadeIn(earth), Write(earth_label))

        self.add(earth)

        # --- Units ---
        for unit in data["units"]:
            x, y = unit["position"]
            z = 0.5  # Default Z height for orbit
            color = self.get_color(unit["allegiance"])
            dot = Dot3D(point=[x, y, z], radius=0.15, color=color)
            label = Text(unit["id"], font_size=10).move_to([x, y, z + 0.3]).scale(0.4)
            group = VGroup(dot, label)
            self.play(FadeIn(group), run_time=0.2)
            self.unit_mobs[unit["id"]] = group

        # --- Objectives ---
        for obj in data.get("objectives", []):
            x, y = obj["location"]
            z = 0.5
            marker = Dot3D(point=[x, y, z], color=YELLOW)
            label = Text(obj["description"], font_size=12).next_to(marker, RIGHT).scale(0.4)
            self.play(FadeIn(marker), Write(label))

        # --- Timeline ---
        for event in data.get("timeline", []):
            desc_text = Text(event["description"], font_size=14).to_edge(DOWN).scale(0.4)
            self.play(Write(desc_text), run_time=0.4)

            for uid in event["involved_units"]:
                mob = self.unit_mobs.get(uid)
                if not mob:
                    continue
                self.play(Indicate(mob), run_time=0.3)

                # Movement
                if event.get("event_type") == "move" and event.get("location"):
                    x, y = event["location"]
                    z = 0.5
                    self.play(mob.animate.move_to([x, y, z]), run_time=1)

                # Fire effect
                elif event.get("event_type") == "fire" and event.get("location"):
                    start = mob.get_center()
                    end = [*event["location"], 0.5]
                    beam = Line3D(start, end, color=RED)
                    pulse = Dot3D(start, color=RED)
                    self.add(beam, pulse)
                    self.play(MoveAlongPath(pulse, beam), run_time=0.6)
                    self.remove(beam, pulse)

            self.remove(desc_text)
            self.wait(0.2)

        self.wait(2)

    def get_color(self, allegiance):
        return {
            "Friendly": BLUE,
            "Home": GREEN,
            "Enemy": RED
        }.get(allegiance, WHITE)