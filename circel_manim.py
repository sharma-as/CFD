from manim import *

class CreateCircle(Scene):
    def construct(self):
        # Create a circle
        circle = Circle()

        # Add the circle to the scene
        self.play(Create(circle))

        # Keep the circle on screen for a while
        self.wait(2)

# To render this scene, save this script as create_circle.py and run the following command in your terminal:
# manim -pql create_circle.py CreateCircle