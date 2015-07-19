from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from math import *
import numpy as np

 


class shooter_l(object):
	def __init__(self, position, angle,init_vel,int_glOrtho,brain):
		self.m_Loc = position;
		self.m_Theta = angle;
		self.m_vel = init_vel;
		self.m_rot = np.zeros((2,2)) ; # rotation matrix
		self.m_range = np.array([0,0,0,0]) ; # range of view
		self.m_bullet_vel = np.array([0 , 0]) ;  # direction of shot
		self.m_limit = int_glOrtho ;
		self.brain = brain ;
		self.m_threat = [] ; #bullet shot at self
		self.m_targets = [] ; # list of shooters in range
		self.m_speed  =  1.5 ; #bullet speed
		self.m_bullet = [];
		b = 0.5 ;
		self.m_boundaries = np.array([[0,0.67*b],[-b,-0.33*b],[b,-0.33*b]])
		
		
	def lookout(self,pos):
		if self.m_threat:
			v1 = self.m_Loc-self.m_threat.b_Loc ;
		else:
			v1 = np.array([0,0]) ; # should not be this think of something else
		c = pos - self.m_Loc;
		m = c.shape[0] ;
		if m < 3:
			c= np.append(c,np.zeros(((3-m),2)),axis=0) ;
		elif m>3:
			sorted_norm = np.argsort(np.apply_along_axis(np.linalg.norm, 1, c))
			c = c[sorted_norm[0:3]] ;
		current_input = np.append(c,np.array([v1]),axis=0).reshape(1,8) ;
		#return np.argmax(self.brain.activate(current_input[0])) ;
		return np.argmax(self.brain.output_single_Vector(current_input)) ;
		
				
				 
		
		
	def update_range(self):
		if(self.m_Theta>0 and self.m_Theta<pi/2):
			self.m_range = np.array([self.m_Loc[0],self.m_limit,self.m_limit,self.m_Loc[1]]) ;
		elif(self.m_Theta>pi/2 and self.m_Theta<pi):
			self.m_range = np.array([ -self.m_limit,self.m_limit,self.m_Loc[0],self.m_Loc[1]]);
		elif(self.m_Theta>pi  and self.m_Theta<(1.5*pi)):
			self.m_range = np.array([-self.m_limit,self.m_Loc[1],self.m_Loc[0],-self.m_limit]);
		else:
			self.m_range = np.array([self.m_Loc[0],self.m_Loc[1],self.m_limit,-self.m_limit]);
	
 

	def draw(self):
		spare = [];
		self.m_rot= np.array([[cos(self.m_Theta-pi/2) , -sin(self.m_Theta-pi/2) ],
		[sin(self.m_Theta-pi/2) , cos(self.m_Theta-pi/2)]]);
		# Compute based upon the current orientation
		spare = np.dot(self.m_rot,self.m_boundaries.T) ;
		glColor3f(0.0,1.0,1.0);
		glBegin(GL_TRIANGLES);
		for i in xrange(0,3):
			glVertex2f(self.m_Loc[0]+spare[0,i],self.m_Loc[1]+spare[1,i]) ;
		glEnd(); 
		glColor3f(1,1,1);
		# Draw the Range
		"""
		glBegin(GL_LINE_LOOP);
		glVertex2f(self.m_range[0],self.m_range[1]) ;
		glVertex2f(self.m_range[0],self.m_range[3]) ;
		glVertex2f(self.m_range[2],self.m_range[3]) ;
		glVertex2f(self.m_range[2],self.m_range[1]) ;
		glEnd();
		"""
		# Draw the bullet if any
		if self.m_bullet:
			self.m_bullet.draw();
	
		

class bullet(object):
	def __init__ (self,pos,vel):
		self.b_Loc = pos ;
		self.b_vel = vel ;

	def draw(self):
		glColor3f(0,0,1);
		glRectf(self.b_Loc[0]-0.1,self.b_Loc[1]+0.1,self.b_Loc[0]+0.1,self.b_Loc[1]-0.1);
	
