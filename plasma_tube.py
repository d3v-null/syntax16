import pyprocessing
from pyprocessing import PVector, color
from math import sqrt, pow, sin, cos, acos, tan, atan, pi, e, log, exp
from random import random, uniform
from numpy import arctan2

screen_size = (320, 200)
back_col = color(0, 0, 0)
default_fill_col = color(255,255,255)
background_particle_count = 100

class Positionable(object):
    """ Mixin for positionable objects """
    def __init__(self, **kwargs):
        self.position = kwargs.get('position', PVector(0,0,0))
        self.orientation = kwargs.get('orientation', PVector(1,0,0))

    @property
    def size(self):
        return self.orientation.mag()

    @property
    def spherical_position(self):
        proj_xy = self.orientation.copy()
        azimuth = PVector(self.orientation.x, self.orientation.y)
        polar = self.orientation.angleBetween(proj_xy)
        return


class Drawable(Positionable):
    """ Mixin for drawable obkects """
    def __init__(self, **kwargs):
        Positionable.__init__(self, **kwargs)
        self.fill_color = kwargs.get('fill_color', default_fill_col)

    def draw_poly(self):
        pass

    def draw(self):
        pyprocessing.pushMatrix()
        pyprocessing.noStroke()
        pyprocessing.fill(self.fill_color)
        pyprocessing.translate(self.position.x, self.position.y, self.position.z)
        self.draw_poly()
        pyprocessing.popMatrix()

class Backgrond_Particle(Drawable):
    def __init__(self, **kwargs):
        Drawable.__init__(self, **kwargs)
        self.active = kwargs.get('active', False)
        self.iteration = 0

    def draw_poly(self):
        pyprocessing.sphere(self.size)

    def draw(self):
        if self.active:
            Drawable.draw(self)

class Spawner(Positionable):
    def __init__(self, **kwargs):
        self.angle = kwargs.get('angle', 0)

    @property
    def spawn_position(self):
        circle = PVector(self.size * sin(self.angle), self.size * cos(self.angle), 0)



def setup():
    global screen_size
    global camera_pos
    global background_particles
    camera_pos = PVector(0,0,max(screen_size))

    #init background_particles
    background_particles = []
    for i in range(1, background_particle_count):
        background_particles.append(
            Backgrond_Particle(
                active=False
            )
        )


def draw():
    global camera_pos

    pyprocessing.camera(
        camera_pos.x, camera_pos.y, camera_pos.z,
        0, 0, 0,
        1, 0, 0
    )

    s = max(screen_size)

    pyprocessing.pointLight(255,255,255,10*s,0,0)
    pyprocessing.pointLight(255,255,255,0,10*s,0)
    pyprocessing.pointLight(255,255,255,0,0,10*s)

    pyprocessing.background(back_col)

    for background_particle in background_particles:
        background_particle.draw()

pyprocessing.run()
