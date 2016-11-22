from __future__ import division
import pyprocessing as pyp

from pyprocessing import PVector, color
from math import sqrt, pow, sin, cos, acos, tan, atan, pi, e, log, exp
from random import random, uniform, seed
import numpy as np
import time
from pprint import pprint, pformat
import os
import errno
import distutils

screen_size = (320, 200)
back_col = color(0, 0, 0, 250)
default_fill_col = color(255,255,255)
background_particle_count = 500
frame_cycles = 360

class Transformation(object):
    """ Classes for transforming PVectors """
    @classmethod
    def transform(cls, transformation, vector):
        assert isinstance(transformation, np.ndarray)
        assert isinstance(vector, PVector)
        # if transformation is 3x3 then vector should be 3
        assert all([len(vector) == dimension for dimension in transformation.shape])

        response = np.matmul(transformation, vector)
        return PVector(*response)

    @classmethod
    def rotate_x(cls, theta, vector):
        assert isinstance(vector, PVector)
        transformation = np.array([[1,           0,            0         ],
                                   [0,           cos(theta), -sin(theta) ],
                                   [0,           sin(theta), cos(theta)  ]])
        return cls.transform(transformation, vector)

    @classmethod
    def rotate_y(cls, theta, vector):
        assert isinstance(vector, PVector)
        transformation = np.array([[cos(theta),  0,          sin(theta)  ],
                                   [0,           1,          0           ],
                                   [-sin(theta), 0,          cos(theta)  ]])
        return cls.transform(transformation, vector)

    @classmethod
    def rotate_z(cls, theta, vector):
        assert isinstance(vector, PVector)
        transformation = np.array([[cos(theta),  -sin(theta), 0           ],
                                   [sin(theta),  cos(theta),  0           ],
                                   [0,           0,           1           ]])
        return cls.transform(transformation, vector)

    @classmethod
    def transpose(cls, offset, vector):
        return PVector(*[sum(elems) for elems in zip(offset, vector)])

class Vector_Helpers(object):
    @classmethod
    def cartesian_to_spherical(cls, vector):
        length = vector.mag()
        azimuth = np.arctan2(vector.x, vector.y)
        polar = cls.angle_between(
            PVector(0,0,1),
            vector
        )
        return (length, polar, azimuth)

    @classmethod
    def angle_between(cls, vector_a, vector_b):
        if vector_a.mag() == 0 or vector_b.mag() == 0:
            return None
        normal_a, normal_b = vector_a.get(), vector_b.get()
        normal_a.normalize()
        normal_b.normalize()

        return np.arccos(
            np.clip(
                normal_a.dot(normal_b),
                -1.0,
                1.0
            )
        )

class Positionable(object):
    """ Mixin for positionable objects """
    def __init__(self, **kwargs):
        self.position = kwargs.get('position', PVector(0,0,0))
        self.orientation = kwargs.get('orientation', PVector(1,0,0))

    @property
    def size(self):
        return self.orientation.mag()

    @property
    def position_spherical(self):
        return Vector_Helpers.cartesian_to_spherical(self.position)

    @property
    def orientation_spherical(self):
        return Vector_Helpers.cartesian_to_spherical(self.orientation)


class Drawable(Positionable):
    """ Mixin for drawable objects """
    def __init__(self, **kwargs):
        Positionable.__init__(self, **kwargs)
        self.fill_color = kwargs.get('fill_color', default_fill_col)
        self.active = kwargs.get('active', True)

    def draw_poly(self):
        pass

    def draw(self):
        pyp.pushMatrix()
        pyp.noStroke()
        pyp.fill(self.fill_color)
        pyp.translate(*self.position)
        self.draw_poly()
        pyp.popMatrix()

class Dynamic(object):
    """ mixin for particles that have move"""
    def __init__(self, **kwargs):
        self.velocity = kwargs.get('velocity', PVector(0,0,0))
        self.mass = kwargs.get('mass', 1)

class Particle(Drawable, Dynamic):
    def __init__(self, **kwargs):
        Drawable.__init__(self, **kwargs)
        Dynamic.__init__(self, **kwargs)
        self.iteration = 0

    def draw_poly(self):
        pyp.sphere(self.size)

        pyp.line(self.position, PVector(0,0,0))

    def draw(self):
        if self.active:
            Drawable.draw(self)

    def step(self, field=None):
        if field:
            self.velocity = PVector(
                *[sum([v_i,f_i]) for (v_i, f_i) in zip(self.velocity, field)]
            )
        if self.velocity:
            self.position = PVector(
                *[sum([p_i, v_i]) for (p_i, v_i) in zip(self.position, self.velocity)]
            )

class Swarm(list, Drawable):
    """ Collection of particles that obey the same rules """
    def __init__(self, **kwargs):
        self.capacity = kwargs.pop('capacity', 100)
        self.spawn_velocity = kwargs.pop('spawn_velocity', PVector(-1,0,0))
        Drawable.__init__(self, **kwargs)
        list.__init__(self, **kwargs)

    def spawn(self, **kwargs):
        # position = kwargs.pop('position')
        vel = kwargs.get('velocity', PVector(0,0,0))
        kwargs['velocity'] = PVector(
            *[sum([s_i, v_i]) for s_i, v_i in zip( self.spawn_velocity, vel) ]
        )
        if len(self) > self.capacity:
            self.pop(0)
        self.append(Particle(
            **kwargs
        ))

class Spawner(Drawable):
    """ An invisible object that moves around is used for spawning particles """
    def __init__(self, **kwargs):
        kwargs['active'] = True
        Drawable.__init__(self, **kwargs)
        self.angle = kwargs.get('angle', 0)
        self.center_particle = Particle(
            active=self.active,
            position=self.position,
            orientation=self.orientation
        )
        self.spawning_particle = Particle(
            active=self.active,
            position=self.spawn_position,
            orientation=self.orientation
        )

    def draw(self):
        self.spawning_particle.position = self.spawn_position
        self.center_particle.draw()
        self.spawning_particle.draw()

    @property
    def spawn_position(self):
        circle = PVector(self.size * sin(self.angle), self.size * cos(self.angle), 0)
        (r, theta, phi) = self.orientation_spherical
        if theta:
            circle = Transformation.rotate_y(-theta, circle)
        if phi:
            circle = Transformation.rotate_z(phi, circle)
        if self.position:
            circle = Transformation.transpose(self.position, circle)
        return circle

def setup():
    """ Required for pyprocessing.run() """
    global screen_size      # size of the raster canvas
    global camera_pos       # PVector of camera's position
    global spawner          # an object used for spawning particles
    global frame_count      # number of framse currently
    global swarm            # Collection of background partciles
    global img_dir          # dir where frames are stored

    run_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    img_dir = './images/%s' % run_stamp
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    frame_count = 0

    s = max(screen_size)
    camera_pos = PVector(0,0,s)
    spawner = Spawner(
        position=PVector(0,0,s),
        orientation=PVector(0,0,s/10)
    )
    swarm = Swarm(
        spawn_velocity=PVector(0,0,-s/40),
        capacity=background_particle_count
    )

def draw():
    """ Required for pyprocessing.run() """
    global screen_size
    global camera_pos
    global spawner
    global frame_count
    global swarm
    global back_col
    global img_dir

    pyp.camera(
        camera_pos.x, camera_pos.y, camera_pos.z,
        0, 0, 0,
        1, 0, 0
    )

    s = max(screen_size)

    pyp.pointLight(255,255,255,10*s,0,0)
    pyp.pointLight(255,255,255,0,10*s,0)
    pyp.pointLight(255,255,255,0,0,10*s)

    pyp.background(back_col)

    #calculate spawner angle
    animation_angle = 2.0 * pi * (frame_count % frame_cycles) / frame_cycles
    spawner.angle = 65 * animation_angle
    # print 'spawner', pformat({
    #     'angle':spawner.angle,
    #     'position':spawner.spawn_position
    # })

    seed(animation_angle)

    swarm.spawn(
        orientation=PVector(s/50,0,0),
        position=spawner.spawn_position,
        velocity=PVector((sin(animation_angle)) * s/100, (cos(animation_angle)) * s/100, (1-random()*2) * s/100),
        active=True
    )

    for particle in swarm:
        # print 'particle', pformat({
        #     'position': particle.position,
        #     'velocity': particle.velocity,
        #     'orientation': particle.orientation
        # })
        particle.draw()
        particle.step()


    frame_count += 1

    if frame_count in range(360,720):
        img = pyp.get()
        # print type(img)
        pyp.save(os.path.join(img_dir, 'image_%s.jpg'%(frame_count)))


pyp.run()
