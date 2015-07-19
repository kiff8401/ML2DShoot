from __future__ import division
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np
import cPickle as pickle
from math import pi ;
from module1_train import *
from module1_ML import *
# not exactly needed
from ann_assign import MLP


 
counter =  0;
possible_vel = np.array([[0,-0.1],[0.1,0],[-0.1,0],[0,0.1]]) ;
possible_theta = np.array([[1.5*pi],[0],[pi],[pi/2]]) ;
n_shooters = 3 ;
data_X = np.zeros((1,8)) #initializing the X_data shape
data_y = np.zeros((1,1))
 
class MainWindow(object):
	def __init__(self,m):
		glutInit();
		self.int_glOrtho = 10 ;
		self.DAMP = 0.25 ;  #DAMPING in computation, more => refreshed often, less => slow refresh
		self.h = 0.3 ;
		self.win_x  = 600;
		self.win_y = 700;
		self.shooters = [];
		self.dt = 0.2 ;
		self.hit_thresh = 0.2 ;
		self.tick_period = 1000 ;
		self.n_decide = 10  ;   # when 50, ~100 when 10 ~400
		self.bdry_thresh = 0.01 ;
		self.brain = m ;
		self.mouse_click=[]
		self.populate();
		
		
	def populate(self):
		# to add more store it in a file and read from it
		
		#self.pos = np.array([[-5,5],[2,-2],[1,2]]) ;
		#self.vel =  np.array([[-0,0.1],[0.1,-0],[0.1,0]]) ;
		#self.theta = np.array([0.6,-pi/2,-pi/2]) ;
		#RANDOM STARTING POINT
		self.pos = np.random.uniform(-0.5*self.int_glOrtho,0.5*self.int_glOrtho,size=(n_shooters,2))
		random_indices = np.int_(np.round(3*np.random.random([n_shooters])))
		self.vel = possible_vel[random_indices]
		self.theta = possible_theta[random_indices] ;
		dummy = []	
		self.mr_anderson = shooter_l(np.array([0,0]),pi,np.array([0,0]),self.int_glOrtho,brain) ;
		for i in xrange(n_shooters):
			self.shooters.append(shooter(self.pos[i],self.theta[i],self.vel[i],self.int_glOrtho,self.mr_anderson));
		
		
	def pre_display(self):
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		self.int_glOrtho = self.win_x/100 ;
		gluOrtho2D(-self.int_glOrtho,self.int_glOrtho,-self.int_glOrtho,self.int_glOrtho);
		
		
	def display(self):
		glClearColor(0.2,0.1,0.1,0.5);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) ;  
		glColor3f(0.2,0.2,0.2);
		self.DrawAxes();
		for shooter in self.shooters:
			shooter.draw();
		if self.mr_anderson:
			self.mr_anderson.draw();
		glutSwapBuffers();
		
		
	
	def open_glut_window(self):
		glutInitDisplayMode ( GLUT_RGBA | GLUT_DOUBLE );
		glutInitWindowPosition ( 0, 80 );
		glutInitWindowSize ( self.win_x, self.win_y );
		glutCreateWindow ( "Killtronics" );

		glClearColor ( 0, 0.,0, 1 );
		glClear ( GL_COLOR_BUFFER_BIT );
		glutSwapBuffers ();
		glEnable(GL_LINE_SMOOTH);
		glEnable(GL_POLYGON_SMOOTH);

		self.pre_display (); 
		glutIdleFunc (self.update);
		glutDisplayFunc (self.display);
	
	def DrawAxes(self):
		for j in np.arange(-self.int_glOrtho,self.int_glOrtho,0.3):
			glBegin(GL_LINES) ;
			glVertex2f(-self.int_glOrtho,j);
			glVertex2f(self.int_glOrtho,j);
			glEnd();
			glBegin(GL_LINES) ;
			glVertex2f(j,-self.int_glOrtho);
			glVertex2f(j,self.int_glOrtho);
			glEnd();

	def  update(self):
		 # DECLARE GLOBAL VARIABLES
		global counter,possible_vel, possible_theta, n_shooters,data_X ,data_y;
		#MONITOR GAME OVER STATUS, all BOTS are KILLED
		if n_shooters == 0 or not self.mr_anderson: 
			print "GAME OVER!"
			glutLeaveMainLoop();
			return ;
		#TAKE CARE OF BOUNDARY CONDITIONS of BOTS
		index = np.unique(np.where(self.int_glOrtho - abs(self.pos) < self.hit_thresh)[0]) ;
		self.vel[index] *=  -1 ;
		# GENERATE RANDOM STATE FOR BOTS EVERY tick_periios iterations
		if counter%self.tick_period == 0:
			random_indices = np.int_(np.round(3*np.random.random([n_shooters])))
			self.vel = possible_vel[random_indices]
			self.theta = possible_theta[random_indices] ;	
		self.pos = self.pos +  (self.dt*self.DAMP)*self.vel ;
		#UPDATE STATUS OF EACH BOT
		i = 0;
		for shooter in self.shooters:
			temp = []
			returned_target = [] ;
			if shooter.m_threat:
				#check if bullet has missed/not in line
				if np.linalg.norm(shooter.m_threat.b_Loc -  shooter.m_Loc) < self.hit_thresh:
					#delete shooter and recording from position,velocity and orientation vectors
					self.shooters.remove(shooter);
					self.pos = np.delete(self.pos,[i],axis=0);
					self.vel = np.delete(self.vel,[i],axis=0);
					self.theta = np.delete(self.theta,[i]);
					n_shooters -= 1 ;
					break;
			#check if  mr_anderson is still alive, only then shoot him
			if self.mr_anderson :
				if self.mr_anderson.m_threat:
					v1 = self.mr_anderson.m_Loc-self.mr_anderson.m_threat.b_Loc ;
					# initial save data point
					v2 = self.mr_anderson.m_threat.b_vel ;
					#v2 = v2/np.linalg.norm(v2);
					if  (v1[1]*v2[1])/(v1[0]*v2[0]) < 0:		#the bullet has passed by, remove threat			 
						self.mr_anderson.m_threat = []
					else:									# in line, check for collison
						if np.linalg.norm(self.mr_anderson.m_threat.b_Loc -  self.mr_anderson.m_Loc) < self.hit_thresh: #its a hit !
							self.mr_anderson = [] ;
							break;
			#always target mr_anderson	
				returned_target  = shooter.lookout();
				if returned_target:
					self.mr_anderson.m_threat = shooter.m_bullet ;
			#update velocities and positions 
			shooter.m_Loc = self.pos[i];
			shooter.m_vel = self.vel[i];
			shooter.m_Theta = self.theta[i] ;
			#print self.DAMP*self.dt*shooter.m_vel
			if shooter.m_bullet:
				shooter.m_bullet.b_Loc = shooter.m_bullet.b_Loc  + self.DAMP*self.dt*shooter.m_bullet.b_vel ;
			i+= 1;
		# EXECUTE NEXT STAT EVERY 'n_decide' ITERATIONS	
		if counter%self.n_decide== 0:
			p = self.mr_anderson.lookout(self.pos);
			self.mr_anderson.m_vel = 2*possible_vel[p] ;
			self.mr_anderson.m_Theta = possible_theta[p] ;
			if  min(self.int_glOrtho - abs(self.mr_anderson.m_Loc))  < self.hit_thresh :
				#print 'true'
				self.mr_anderson.m_vel *= -1 ;
			#SHOULD NOT BE DOING THE EXTRA 2 !!
		counter+= 1 ; 
		if self.mr_anderson:
				self.mr_anderson.m_Loc = self.mr_anderson.m_Loc  + self.DAMP*self.dt*self.mr_anderson.m_vel
				if self.mr_anderson.m_bullet:
					self.mr_anderson.m_bullet.b_Loc = self.mr_anderson.m_bullet.b_Loc  + self.DAMP*self.dt*self.mr_anderson.m_bullet.b_vel ;
		glutPostRedisplay ();
		
	def StartAnimation(self):
		self.open_glut_window();
		glutMainLoop();

if __name__ == '__main__': 
	#parser = ap.ArgumentParser(description='argument for save file name') ; 
	#load the brain
	brain = pickle.load(open("brain_v1.p","rb"))
	#brain = pickle.load(open("brain_v2.p","rb"))
	window = MainWindow(brain);
	window.StartAnimation();
	 
	
		
