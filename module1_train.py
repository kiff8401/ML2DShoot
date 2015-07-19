from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from math import *
import numpy as np

 
 


class shooter(object):
	def __init__(self, position, angle,init_vel,int_glOrtho,human):
		self.m_Loc = position;
		self.m_Theta = angle;
		self.m_vel = init_vel;
		self.m_rot = np.zeros((2,2)) ; # rotation matrix
		self.m_range = np.array([0,0,0,0]) ; # range of view
		self.m_limit = int_glOrtho ;
		self.m_threat = [] ; #bullet shot at self
		self. m_target  = human ;
		self.m_bullet = [];
		self.m_speed = 1 ; #the speed of the bullet fired
		b = 0.5 ;
		self.m_boundaries = np.array([[0,0.67*b],[-b,-0.33*b],[b,-0.33*b]])
		self.update_range();
		#self.lookout(); # watchout from the moment you exist!
		
	def lookout(self):
		# set rotation matrix
		
		# set the vision range
		self.update_range();
		# avoid overflow in theta	
		if(self.m_Theta>(2*pi)):
			self.m_Theta = 0.01 ;
		# keep rotating to  keep looking around
		#self.m_Theta += pi/1000;
		# shoot at target
		if not self.m_target.m_threat:
			v = self.m_target.m_Loc - self.m_Loc;
			self.m_bullet = bullet(self.m_Loc.copy(),self.m_speed*(v/np.linalg.norm(v))) ;
			return self.m_target	
		
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
		glColor3f(1.0,1.0,1.0);
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
	
