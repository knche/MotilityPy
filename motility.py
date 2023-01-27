import numpy
import math
import smallestenclosingcircle
import scipy.spatial
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import welch
from scipy.stats import binned_statistic

class motility :

	def __init__ ( self, data= [], micron_px = 1, time_seq = 1, time_acquisition = 0 ):
		self.data = data
		self.micron_px = micron_px
		self.time_seq = time_seq
		self.time_adquisition = time_acquisition
		self.frame = numpy.full( ( time_acquisition, len(self.data), 25 ), numpy.nan )
		self.divison_time = []
		self.cell_size = []
		self.aspect_ratio = []
		self.growth_rate = []
		self.shift_pos = numpy.full( ( self.frame.shape[0], self.frame.shape[1], 4 ), numpy.nan )
		self.shift_corr = numpy.full( ( self.frame.shape[0], self.frame.shape[1], 3 ), numpy.nan )
		self.shift_tamsd = numpy.full( (self.frame.shape[0], self.frame.shape[1], 2 ), numpy.nan )
		self.shift_tamme = numpy.full( (self.frame.shape[0], self.frame.shape[1] ), numpy.nan )
		self.shift_persistence_distance = numpy.full( (self.frame.shape[0], self.frame.shape[1], 3), numpy.nan )
		self.shift_angle = numpy.full( (self.frame.shape[0], self.frame.shape[1], 2 ), numpy.nan )
				
		self.frame_global = numpy.full( ( len(self.data), 60 ), numpy.nan )

		self.g_speed = 0
		self.std_g_speed = 0

		self.graph = self.resource_graph()

	def run ( self ):

		# x, y, angle, view angle
		for i in range( len(self.data) ):
			for j in range( len(self.data[i]) ):
				self.frame[self.data[i][j][0],i,0] = self.data[i][j][3]*self.micron_px
				self.frame[self.data[i][j][0],i,1] = self.data[i][j][4]*self.micron_px
				self.frame[self.data[i][j][0],i,10] = self.data[i][j][5]
				self.frame[self.data[i][j][0],i,22] = self.data[i][j][1]*self.micron_px
				if j > 0 :
					viewtheta = math.atan( (self.data[i][j][4] - self.data[i][0][4])/(self.data[i][j][3] - self.data[i][0][3]) )*180/math.pi
					self.frame[self.data[i][j-1][0],i,15] = viewtheta
					
					if self.data[i][j][3] > self.data[i][0][3] and self.data[i][j][4] > self.data[i][0][4]:
						viewtheta = viewtheta
					elif self.data[i][j][3] > self.data[i][0][3] and self.data[i][j][4] < self.data[i][0][4]:
						viewtheta = 360 + viewtheta
					elif self.data[i][j][3] < self.data[i][0][3] and self.data[i][j][4] > self.data[i][0][4]:
						viewtheta = 180 + viewtheta
					elif self.data[i][j][3] < self.data[i][0][3] and self.data[i][j][4] < self.data[i][0][4]:
						viewtheta = 180 + viewtheta
					
					self.frame[self.data[i][j-1][0],i,16] = viewtheta

				# turning angle
				if j > 0 and j < (len(self.data[i])-1):
					fdX = (self.data[i][j][3] - self.data[i][j-1][3])*self.micron_px
					fdY = (self.data[i][j][4] - self.data[i][j-1][4])*self.micron_px
					fdistance = math.sqrt( math.pow( fdX, 2 ) + math.pow( fdY, 2 ) )

					ldX = (self.data[i][j+1][3] - self.data[i][j][3])*self.micron_px
					ldY = (self.data[i][j+1][4] - self.data[i][j][4])*self.micron_px
					ldistance = math.sqrt( math.pow( ldX, 2 ) + math.pow( ldY, 2 ) )				

					self.frame[self.data[i][j][0],i,21] = math.acos( round( ( fdX*ldX + fdY*ldY )/(fdistance*ldistance), 2) )*180/math.pi


		# deltaX, deltaY, distance, velocity x, velocity x, instant velocity
		self.frame[1:self.time_adquisition,:,2] = self.frame[1:self.time_adquisition,:,0] - self.frame[0:self.time_adquisition-1,:,0]
		self.frame[1:self.time_adquisition,:,3] = self.frame[1:self.time_adquisition,:,1] - self.frame[0:self.time_adquisition-1,:,1]
		self.frame[:,:,4] = numpy.sqrt( numpy.power( self.frame[:,:,2], 2) + numpy.power( self.frame[:,:,3], 2) )
		self.frame[:,:,5] = self.frame[:,:,2]/self.time_seq
		self.frame[:,:,6] = self.frame[:,:,3]/self.time_seq
		self.frame[:,:,7] = self.frame[:,:,4]/self.time_seq

		# time averaged msd
		for i in range( len(self.data) ):
			for j in range( len(self.data[i]) ):
				sum_tamsd = 0
				sum_tamsd_4 = 0
				sum_tamsd_x = 0
				sum_tamsd_y = 0
				for k in range( 0, len(self.data[i]) - j ):
					sum_tamsd = sum_tamsd + math.pow(self.data[i][k+j][3]*self.micron_px - self.data[i][k][3]*self.micron_px,2) + math.pow(self.data[i][k+j][4]*self.micron_px - self.data[i][k][4]*self.micron_px,2)
					sum_tamsd_4 = sum_tamsd + math.pow(self.data[i][k+j][3]*self.micron_px - self.data[i][k][3]*self.micron_px,4) + math.pow(self.data[i][k+j][4]*self.micron_px - self.data[i][k][4]*self.micron_px,4)
					sum_tamsd_x = sum_tamsd_x + math.pow(self.data[i][k+j][3]*self.micron_px - self.data[i][k][3]*self.micron_px,2)
					sum_tamsd_y = sum_tamsd_y + math.pow(self.data[i][k+j][4]*self.micron_px - self.data[i][k][4]*self.micron_px,2)
				self.frame[self.data[i][j][0],i,8] = (sum_tamsd)/( len(self.data[i]) - j )
				self.frame[self.data[i][j][0],i,18] = (sum_tamsd_x)/( len(self.data[i]) - j )
				self.frame[self.data[i][j][0],i,19] = (sum_tamsd_y)/( len(self.data[i]) - j )
				self.frame[self.data[i][j][0],i,20] = (sum_tamsd_4)/( len(self.data[i]) - j )

		# time average mme
		for i in range( len(self.data) ):
			for j in range( len(self.data[i]) ):
				sum_tamme = 0
				for k in range( 0, len(self.data[i]) - j ):
					xmax = []
					for z in range(k,k+j+1):
						xmax.append( math.pow(self.data[i][z][3]*self.micron_px - self.data[i][k][3]*self.micron_px,2) + math.pow(self.data[i][z][4]*self.micron_px - self.data[i][k][4]*self.micron_px,2) )
					sum_tamme = sum_tamme + numpy.nanmax( xmax )		
				self.frame[self.data[i][j][0],i,17] = (sum_tamme)/( len(self.data[i]) - j )		

		# persistance distance & fmi
		for i in range( len(self.data) ):
			for j in range( 1, len(self.data[i]) ):
				sum_path = 0
				distance = math.sqrt( math.pow(self.data[i][j][3] - self.data[i][0][3],2) + math.pow(self.data[i][j][4] - self.data[i][0][4],2) )*self.micron_px
				distx = (self.data[i][j][3] - self.data[i][0][3])*self.micron_px
				disty = (self.data[i][j][4] - self.data[i][0][4])*self.micron_px
				for k in range(1,j+1):
					sum_path = sum_path + math.sqrt( math.pow(self.data[i][k][3] - self.data[i][k-1][3],2) + math.pow(self.data[i][k][4] - self.data[i][k-1][4],2) )*self.micron_px
				
				self.frame[self.data[i][j][0],i,9] = distance/sum_path if sum_path > 0 else 0
				self.frame[self.data[i][j][0],i,13] = distx/sum_path if sum_path > 0 else 0
				self.frame[self.data[i][j][0],i,14] = disty/sum_path if sum_path > 0 else 0

		# correlation speed
		for i in range( len(self.data) ):
			for j in range( len(self.data[i]) ):
				sum_corr = 0
				avg_speed = numpy.nanmean( numpy.power( self.frame[:,i,7],2 ) )
				for k in range( 1, len(self.data[i]) - j ):
					dx = (self.data[i][k-1][3]-self.data[i][k][3])*(self.data[i][k+j-1][3]-self.data[i][k+j][3])*( math.pow(self.micron_px,2) )
					dy = (self.data[i][k-1][4]-self.data[i][k][4])*(self.data[i][k+j-1][4]-self.data[i][k+j][4])*( math.pow(self.micron_px,2) )
					sum_corr = sum_corr + ((dx+dy)/( math.pow(self.time_seq,2) ))/avg_speed
				self.frame[self.data[i][j][0],i,11] = (sum_corr)/( len(self.data[i]) - j )


		#division time, cell size, growth rate & aspect ratio
		for path in self.data:
			if path[0][2] > 0 :
				self.aspect_ratio.append( path[0][1]/path[0][2] )
			else:
				self.aspect_ratio.append( numpy.nan )

			self.divison_time.append( len(path)*self.time_seq - self.time_seq )
			self.cell_size.append( path[0][1]*self.micron_px )
			self.growth_rate.append( abs(path[ len(path) -1 ][1] - path[0][1])*self.micron_px/(len(path)*self.time_seq - self.time_seq) )

		#persistence time
		#self.frame[:,:,12] = numpy.full( self.frame[:,:,12].shape, numpy.nan )
		'''per25_t_div = numpy.percentile(self.divison_time, 25)/self.time_seq
		per75_t_div = numpy.percentile(self.divison_time, 75)/self.time_seq
		per50_t_div = numpy.percentile(self.divison_time, 50)/self.time_seq
		for i in range( self.frame.shape[1] ):
			i_nonnan = numpy.where( ~numpy.isnan(self.frame[:,i,10]) )[0]
			if len(i_nonnan) >= per50_t_div and len(i_nonnan) <= (per75_t_div + 1.5*(per75_t_div-per25_t_div)) :
				xinit_position = i_nonnan[0]
				self.frame[xinit_position,i,12] = 1
				self.frame[i_nonnan[ len(i_nonnan) - 1 ],i,12] = 1
				for j in i_nonnan[1:]:
					if abs( self.frame[j,i,10] - self.frame[xinit_position,i,10] ) >= 90:
						xinit_position = j
						self.frame[xinit_position,i,12] = 1'''
		
		self.frame[ self.frame[:,:,21]>= 90 , 12] = 1

		# run rollXYDistance
		self.rollXYDistance()
		self.rollVelocityCorrelation()
		self.rollTAMSD()
		self.rollTAMME()
		self.rollPersistenceDistance()
		self.rollAngle()

		# DATA GLOBAL
		
		for i in range( len(self.data) ):
			points = []
			path = 0
			for j in range( len(self.data[i]) ):
				points.append([ self.data[i][j][3]*self.micron_px, self.data[i][j][4]*self.micron_px ])
				if j > 0:
					dX = (self.data[i][j][3] - self.data[i][j-1][3])*self.micron_px
					dY = (self.data[i][j][4] - self.data[i][j-1][4])*self.micron_px
					path = path + math.sqrt( math.pow( dX, 2 ) + math.pow( dY, 2 ) )
			
			xcircle = smallestenclosingcircle.make_circle(points)
			max_dist = numpy.nanmax( scipy.spatial.distance.pdist( points,'euclidean') )
			deltaY = (self.data[i][len(self.data[i])-1][4] - self.data[i][0][4])*self.micron_px
			deltaX = (self.data[i][len(self.data[i])-1][3] - self.data[i][0][3])*self.micron_px
			distance = math.sqrt( math.pow( deltaX, 2 ) + math.pow( deltaY, 2 ) )
						
			self.frame_global[i,0] = xcircle[0] # x origin blob
			self.frame_global[i,1] = xcircle[1] # y origin blob
			self.frame_global[i,2] = xcircle[2]*2 # blob diameter ()
			self.frame_global[i,3] = max_dist # max distance (maximum distance traveled)
			self.frame_global[i,4] = path # sum all step (total distance traveled)
			self.frame_global[i,5] = deltaX # x end-to-end-distance
			self.frame_global[i,6] = deltaY # y end-to-end distance
			self.frame_global[i,7] = distance # end-to-end distance (net distance traveled)
			self.frame_global[i,8] = numpy.nan if path <= 0 else distance / path # persistance (confinement ratio or persistence ratio)
			self.frame_global[i,9] = abs( deltaX / deltaY ) # abs( x/y ) ratio (directionality ratio)
			self.frame_global[i,10] = numpy.nan if path <= 0 else deltaX / path # FMIx (x forward migration index)
			self.frame_global[i,11] = numpy.nan if path <= 0 else deltaY / path # FMIy (y forward migration index)
			self.frame_global[i,12] = distance / (len(self.data[i])*self.time_seq - self.time_seq ) # average speed (mean straight-line speed)
			self.frame_global[i,13] = abs( deltaX ) / (len(self.data[i])*self.time_seq - self.time_seq ) # average speed X (x mean straight-line speed)
			self.frame_global[i,14] = abs( deltaY ) / (len(self.data[i])*self.time_seq - self.time_seq ) # average speed Y (y mean straight-line speed)
			self.frame_global[i,15] = path / (len(self.data[i])*self.time_seq - self.time_seq ) # instant speed (total speed)

			i_nonnan = numpy.where( ~numpy.isnan(self.frame[:,i,7]) )[0]
			self.frame_global[i,16] = numpy.sum( self.frame[i_nonnan,i,7] ) /len(i_nonnan) # ( mean curvilinear speed )

			self.frame_global[i,17] = self.frame_global[i,12] / self.frame_global[i,16] # (linearity of forward progression)
			self.frame_global[i,18] = distance / max_dist # (displacement ratio)
			self.frame_global[i,19] = numpy.nan if path <= 0 else max_dist / path # (Outreach ratio)
			self.frame_global[i,20] =  numpy.nan if path <= 0 else xcircle[2]*2 / path  # (Explorer ratio)
			
			self.frame_global[i,21] = math.nan
			self.frame_global[i,22] = math.nan
			self.frame_global[i,23] = math.nan
			if len(points) > 2 :
				try:
					hull = scipy.spatial.ConvexHull( numpy.array(points) )
					self.frame_global[i,21] = hull.area
					self.frame_global[i,22] = hull.volume
					self.frame_global[i,23] = math.pow( hull.area, 2)/( 4*math.pi*hull.volume )
				except:
					self.frame_global[i,21] = math.nan
					self.frame_global[i,22] = math.nan
					self.frame_global[i,23] = math.nan
				
			dtheta = math.atan( deltaY/deltaX )*180/math.pi
			if self.data[i][len(self.data[i])-1][3] > self.data[i][0][3] and self.data[i][len(self.data[i])-1][4] > self.data[i][0][4] :
				dtheta = dtheta
			elif self.data[i][len(self.data[i])-1][3] > self.data[i][0][3] and self.data[i][len(self.data[i])-1][4] < self.data[i][0][4] :
				dtheta = 360 + dtheta
			elif self.data[i][len(self.data[i])-1][3] < self.data[i][0][3] and self.data[i][len(self.data[i])-1][4] > self.data[i][0][4] :
				dtheta = 180 + dtheta
			elif self.data[i][len(self.data[i])-1][3] < self.data[i][0][3] and self.data[i][len(self.data[i])-1][4] < self.data[i][0][4] :
				dtheta = 180 + dtheta

			self.frame_global[i,24] = dtheta # global mass center orientation
			if len(self.data[i][0]) > 6:
				self.frame_global[i,25] = self.data[i][0][6] # cell body orientation first
			else:
				self.frame_global[i,25] = math.nan

			# view angle
			
			i_nonnan = numpy.where( ~numpy.isnan(self.frame[:,i,16]) )[0]
			mcomb = numpy.nonzero( numpy.tri( len(i_nonnan) , len(i_nonnan)  , -1) )
			mpair = numpy.dstack( (mcomb[1],mcomb[0]) )[0]
			globalviewangle = []
			for row in mpair:
				if self.frame[ i_nonnan[ row[1] ] ,i,16] >= 270 and self.frame[ i_nonnan[ row[0] ] ,i,16] <= 90 :
					globalviewangle.append( ( 360 - self.frame[ i_nonnan[ row[1] ] ,i,16] ) + self.frame[ i_nonnan[ row[0] ] ,i,16] )
				elif self.frame[ i_nonnan[ row[0] ] ,i,16] >= 270 and self.frame[ i_nonnan[ row[1] ] ,i,16] <= 90 :
					globalviewangle.append( ( 360 - self.frame[ i_nonnan[ row[0] ] ,i,16] ) + self.frame[ i_nonnan[ row[1] ] ,i,16] )
				else:
					globalviewangle.append( self.frame[ i_nonnan[ row[1] ] ,i,16] - self.frame[ i_nonnan[ row[0] ] ,i,16] )

			self.frame_global[i,26] = numpy.amax( globalviewangle ) if len(globalviewangle) > 0 else math.nan

			if self.frame[ i_nonnan[ len(i_nonnan) - 1 ] ,i,16] >= 270 and self.frame[ i_nonnan[0] ,i,16] <= 90 :
				self.frame_global[i,27] = -( ( 360 - self.frame[ i_nonnan[ len(i_nonnan) - 1 ] ,i,16] ) + self.frame[ i_nonnan[0] ,i,16] )
			elif self.frame[ i_nonnan[0] ,i,16] >= 270 and self.frame[ i_nonnan[ len(i_nonnan) - 1 ] ,i,16] <= 90 :
				self.frame_global[i,27] = ( 360 - self.frame[ i_nonnan[0] ,i,16] ) + self.frame[ i_nonnan[ len(i_nonnan) - 1 ] ,i,16]
			else:
				self.frame_global[i,27] = self.frame[ i_nonnan[ len(i_nonnan) - 1 ] ,i,16] - self.frame[ i_nonnan[0] ,i,16]
			

			'''
			mcomb = numpy.nonzero( numpy.tri( len(self.data[i]) - 1 , len(self.data[i]) - 1  , -1) )
			mpair = numpy.dstack( (mcomb[1],mcomb[0]) )[0]
			globalviewangle = []
			for row in mpair:
				
				fdX = (self.data[i][ row[0] + 1 ][3] - self.data[i][0][3])*self.micron_px
				fdY = (self.data[i][ row[0] + 1 ][4] - self.data[i][0][4])*self.micron_px
				fdistance = math.sqrt( math.pow( fdX, 2 ) + math.pow( fdY, 2 ) )

				ldX = (self.data[i][ row[1] + 1  ][3] - self.data[i][0][3])*self.micron_px
				ldY = (self.data[i][ row[1] + 1  ][4] - self.data[i][0][4])*self.micron_px
				ldistance = math.sqrt( math.pow( ldX, 2 ) + math.pow( ldY, 2 ) )				

				globalviewangle.append( math.acos( round( ( fdX*ldX + fdY*ldY )/(fdistance*ldistance), 2) )*180/math.pi )

			self.frame_global[i,26] = numpy.amax( globalviewangle )


			if len(self.data[i]) >= 2:

				fdX = (self.data[i][1][3] - self.data[i][0][3])*self.micron_px
				fdY = (self.data[i][1][4] - self.data[i][0][4])*self.micron_px
				fdistance = math.sqrt( math.pow( fdX, 2 ) + math.pow( fdY, 2 ) )

				ldX = (self.data[i][ len(self.data[i]) -1  ][3] - self.data[i][0][3])*self.micron_px
				ldY = (self.data[i][ len(self.data[i]) -1  ][4] - self.data[i][0][4])*self.micron_px
				ldistance = math.sqrt( math.pow( ldX, 2 ) + math.pow( ldY, 2 ) )

				xangle = math.acos( ( fdX*ldX + fdY*ldY )/(fdistance*ldistance) )*180/math.pi
				if ldY < fdY :
					xangle = -xangle

				self.frame_global[i,27] = xangle
			'''			

			Txx = numpy.average( numpy.power( numpy.array(points)[0,:], 2) ) - math.pow( numpy.average( numpy.array(points)[0,:] ) ,2)
			Tyy = numpy.average( numpy.power( numpy.array(points)[1,:], 2) ) - math.pow( numpy.average( numpy.array(points)[1,:] ) ,2)
			Txy = numpy.average( numpy.array(points)[0,:]*numpy.array(points)[1,:] ) - numpy.average(numpy.array(points)[0,:])*numpy.average(numpy.array(points)[1,:])

			self.frame_global[i,30] = Txx # Txx gyration tensor
			self.frame_global[i,31] = Tyy # Tyy gyration tensor
			self.frame_global[i,32] = Txy # Txy gyration tensor
			self.frame_global[i,33] = math.atan( 2*Txy/(Txx-Tyy) )/2 # gyration angle
			self.frame_global[i,34] = ( (Txx+Tyy) + math.sqrt( math.pow(Txx-Tyy,2) + 4*math.pow(Txy,2) ) )/2 # eigenvalue R1^2
			self.frame_global[i,35] = ( (Txx+Tyy) - math.sqrt( math.pow(Txx-Tyy,2) + 4*math.pow(Txy,2) ) )/2 # eigenvalue R2^2

			self.frame_global[i,36] = math.log(len(self.data[i])-1)/math.log((len(self.data[i])-1)*max_dist/path) if math.log((len(self.data[i])-1)*max_dist/path) > 0 else 1 # fractal dimension, 1 straight trajectories, 2 random, 3 constrained

			self.frame_global[i,39] = len(points) # count points
			self.frame_global[i,40] = self.data[i][0][0]*self.time_seq # start time
			self.frame_global[i,41] = self.data[i][ len(self.data[i]) - 1 ][0]*self.time_seq # end time

			self.frame_global[i,42] = self.data[i][0][1] # start time -> major axis
			self.frame_global[i,43] = self.data[i][0][2] # start time -> minor axis

			self.frame_global[i,44] = self.data[i][len(self.data[i]) - 1][1] # end time -> major axis
			self.frame_global[i,45] = self.data[i][len(self.data[i]) - 1][2] # end time -> minor axis

			self.frame_global[i,50] = self.data[i][0][0] # start index
			self.frame_global[i,51] = self.data[i][ len(self.data[i]) - 1 ][0] # end index

		self.__refreshGraph()


	# dtype = path_length, blob, end_to_end_distance, max_distance, time, fractal
	# operation = range, equal, greater, smaller, interquartile1.5x, 1.5xonlythirdquartile, interquartile, notequaltoacquisition, greateracquisition
	# ft = percentile, bias
	def filter ( self, dtype = '', ft = '', operation = 'range', start_value = 25, end_value = 0, offset_time = 10 ):
		
		data = []
		pct_high = 0
		pct_low = 0
		index = []
		if ft == 'percentile' and dtype == 'path_length':
			data = self.getGlobalPath()
		elif ft == 'percentile' and dtype == 'blob':
			data = self.getGlobalDiameterBlob()
		elif ft == 'percentile' and dtype == 'end_to_end_distance':
			data = self.getGlobalEndtoEndDistance()
		elif ft == 'percentile' and dtype == 'maxdistance':
			data = self.getGlobalMaxDistance()

		if ft == 'percentile' and operation == 'range':
			pct_high = numpy.nanpercentile( data, end_value )
			pct_low = numpy.nanpercentile( data, start_value )
		elif ft == 'percentile' and operation == 'equal':
			pct_high = numpy.nanpercentile( data, start_value )
		elif ft == 'percentile' and operation == 'greater':
			pct_high = numpy.nanpercentile( data, start_value )
		elif ft == 'percentile' and operation == 'smaller':
			pct_high = numpy.nanpercentile( data, start_value )
		elif ft == 'percentile' and operation == 'interquartile1.5x':
			pct_high = numpy.nanpercentile( data, 75 )
			pct_low = numpy.nanpercentile( data, 25 )
		elif ft == 'percentile' and operation == '1.5xonlythirdquartile':
			pct_high = numpy.nanpercentile( data, 75 )
			pct_low = numpy.nanpercentile( data, 25 )
		elif ft == 'percentile' and operation == 'interquartile':
			pct_high = numpy.nanpercentile( data, 75 )
			pct_low = numpy.nanpercentile( data, 25 )


		if ft == 'percentile' and operation == 'range':
			index = numpy.where( numpy.all( [data >= pct_low, data <= pct_high], axis = 0 ) )[0]
		elif ft == 'percentile' and operation == 'equal':
			index = numpy.where( data == pct_high )[0]
		elif ft == 'percentile' and operation == 'greater':
			index = numpy.where( data >= pct_high )[0]
		elif ft == 'percentile' and operation == 'smaller':
			index = numpy.where( data <= pct_high )[0]
		elif ft == 'percentile' and operation == 'interquartile1.5x':
			index = numpy.where( numpy.all( [data >= (pct_low - 1.5*(pct_high - pct_low)), data <= ( pct_high + 1.5*(pct_high - pct_low) ) ], axis = 0 ) )[0]
		elif ft == 'percentile' and operation == '1.5xonlythirdquartile':
			index = numpy.where( numpy.all( [data >= pct_low, data <= ( pct_high + 1.5*(pct_high - pct_low) ) ], axis = 0 ) )[0]
		elif ft == 'percentile' and operation == 'interquartile':
			index = numpy.where( numpy.all( [data >= pct_low, data <= pct_high ], axis = 0 ) )[0]
		elif dtype == 'time' and operation == 'notequaltoacquisition':
			xsum = numpy.sum( ~numpy.isnan( self.frame[:,:,0] ), axis = 0 )
			index = numpy.where( xsum < (self.frame.shape[0] - offset_time) )[0]
		elif dtype == 'time' and operation == 'greateracquisition':
			xsum = numpy.sum( ~numpy.isnan( self.frame[:,:,0] ), axis = 0 )
			index = numpy.where( xsum > offset_time )[0]
		elif dtype == 'fractal' and operation == 'smaller':
			index = numpy.where( self.getGlobalFractalDimension() < start_value )[0]
		elif dtype == 'fractal' and operation == 'notzero':
			index = numpy.where( self.getGlobalFractalDimension() > 0 )[0]
		

		self.frame = self.frame[:,index,:]
		self.shift_pos = self.shift_pos[:,index,:]
		self.shift_corr = self.shift_corr[:,index,:]
		self.shift_tamsd = self.shift_tamsd[:,index,:]
		self.shift_tamme = self.shift_tamme[:,index]
		self.shift_persistence_distance = self.shift_persistence_distance[:,index,:]
		self.shift_angle = self.shift_angle[:,index,:]
		self.frame_global = self.frame_global[index,:]

		self.divison_time = numpy.array(self.divison_time)[index].tolist()
		self.cell_size = numpy.array( self.cell_size )[index].tolist()
		self.aspect_ratio = numpy.array( self.aspect_ratio )[index].tolist()
		self.growth_rate = numpy.array( self.growth_rate )[index].tolist()

		self.__refreshGraph()

		return index


	def rollXYDistance( self ) :
				
		for i in range( self.frame.shape[1] ):
			i_nonnan = numpy.where( ~numpy.isnan(self.frame[:,i,0]) )[0]
			self.shift_pos[:,i,0] = numpy.roll(self.frame[:,i,0], self.frame.shape[0] - i_nonnan[0] ) - self.frame[i_nonnan[0],i,0]
			self.shift_pos[:,i,1] = numpy.roll(self.frame[:,i,1], self.frame.shape[0] - i_nonnan[0] ) - self.frame[i_nonnan[0],i,1]
			self.shift_pos[:,i,2] = numpy.roll(self.frame[:,i,4], self.frame.shape[0] - i_nonnan[0] )
			self.shift_pos[:,i,3] = numpy.nancumsum( self.shift_pos[:,i,2] )
			self.shift_pos[ self.shift_pos[:,i,3] == 0 ,i,3] = numpy.nan

	def rollVelocityCorrelation ( self ):
		
		for i in range( self.frame.shape[1] ):
			i_nonnan = numpy.where( ~numpy.isnan(self.frame[:,i,11]) )[0]
			self.shift_corr[:,i,0] = numpy.roll(self.frame[:,i,11], self.frame.shape[0] - i_nonnan[0] )
			self.shift_corr[0:(i_nonnan[len(i_nonnan)-1] - i_nonnan[0] + 1),i,1] = numpy.fft.fft( self.frame[i_nonnan[0]:i_nonnan[ len(i_nonnan) -1 ]+1,i,11] ).real
			self.shift_corr[0:(i_nonnan[len(i_nonnan)-1] - i_nonnan[0] + 1),i,2] = numpy.fft.fft( self.frame[i_nonnan[0]:i_nonnan[ len(i_nonnan) -1 ]+1,i,11] ).imag

	def rollTAMSD ( self ):

		for i in range(self.frame.shape[1]):
			xlen = len(self.frame[~numpy.isnan(self.frame[:,i,8]),i,8])
			self.shift_tamsd[0:xlen,i,0] = self.frame[~numpy.isnan(self.frame[:,i,8]),i,8]
			self.shift_tamsd[0:xlen,i,1] = self.frame[~numpy.isnan(self.frame[:,i,20]),i,20]

	def rollTAMME ( self ):

		for i in range(self.frame.shape[1]):
			xlen = len(self.frame[~numpy.isnan(self.frame[:,i,17]),i,17])
			self.shift_tamme[0:xlen,i] = self.frame[~numpy.isnan(self.frame[:,i,17]),i,17]

	def rollPersistenceDistance ( self ):

		for i in range(self.frame.shape[1]):
			xlen = len(self.frame[~numpy.isnan(self.frame[:,i,9]),i,9])
			self.shift_persistence_distance[0:xlen,i,0] = self.frame[~numpy.isnan(self.frame[:,i,9]),i,9]
			self.shift_persistence_distance[0:xlen,i,1] = self.frame[~numpy.isnan(self.frame[:,i,13]),i,13]
			self.shift_persistence_distance[0:xlen,i,2] = self.frame[~numpy.isnan(self.frame[:,i,14]),i,14]

	def rollAngle ( self ) :

		for i in range( self.frame.shape[1] ):
			i_nonnan = numpy.where( ~numpy.isnan(self.frame[:,i,15]) )[0]
			self.shift_angle[:,i,0] = numpy.roll(self.frame[:,i,15], self.frame.shape[0] - i_nonnan[0] )

	def computeGSpeed ( self, explore_fit = [] ):
		if type( explore_fit ) is list:
			xgspeed = []
			for fit in explore_fit:
				xgspeed.append( ( fit-1/numpy.average(self.divison_time) )*numpy.average(self.cell_size) )
			self.g_speed = numpy.average( xgspeed )
			self.std_g_speed = numpy.std( xgspeed )
		else:
			self.g_speed = ( explore_fit-1/numpy.average(self.divison_time) )*numpy.average(self.cell_size)

	def setFitTAMSD ( self, window = (0,6) ):

		def fit_tamsd(t, alpha, org):
			return alpha*t + org
			
		time_tamsd = numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq

		for i in range( self.shift_tamsd.shape[1] ):
			
			split_time = time_tamsd[ (window[0]+1):window[1] ]
			split_average = self.shift_tamsd[ (window[0]+1):window[1],i,0 ]
			i_nonnan = numpy.where( ~numpy.isnan(split_average) )[0]
			split_time = split_time[i_nonnan]
			split_average = split_average[i_nonnan]


			xrest = curve_fit( fit_tamsd, numpy.log10(split_time), numpy.log10(split_average) )
			
			scaling_exponent_1 = ( math.log10(self.shift_tamsd[2,i,0]) - math.log10(self.shift_tamsd[1,i,0]) )/(math.log10( time_tamsd[2] ) - math.log10( time_tamsd[1] ))
			coefficient_diffusion_1 = math.pow(10,math.log10(self.shift_tamsd[1,i,0]) - scaling_exponent_1*math.log10( time_tamsd[1] ))

			self.frame_global[i,28] = xrest[0][0]
			self.frame_global[i,29] = math.pow(10, xrest[0][1])

			self.frame_global[i,37] = scaling_exponent_1
			self.frame_global[i,38] = coefficient_diffusion_1

	def setFitTAMME ( self, window = (0,6) ):

		def fit_tamme(t, alpha, org):
			return alpha*t + org
			
		time_tamme = numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq

		for i in range( self.shift_tamme.shape[1] ):
			
			split_time = time_tamme[ (window[0]+1):window[1] ]
			split_average = self.shift_tamme[ (window[0]+1):window[1],i ]
			xrest = curve_fit( fit_tamme, numpy.log10(split_time), numpy.log10(split_average) )
			
			scaling_exponent_1 = ( math.log10(self.shift_tamme[2,i]) - math.log10(self.shift_tamme[1,i]) )/(math.log10( time_tamme[2] ) - math.log10( time_tamme[1] ))
			coefficient_diffusion_1 = math.pow(10,math.log10(self.shift_tamme[1,i]) - scaling_exponent_1*math.log10( time_tamme[1] ))

			self.frame_global[i,46] = xrest[0][0]
			self.frame_global[i,47] = math.pow(10, xrest[0][1])

			self.frame_global[i,48] = scaling_exponent_1
			self.frame_global[i,49] = coefficient_diffusion_1

	# getters Details

	def getAllDeltaX ( self, is_flat = False ):
		if is_flat :
			return self.frame[:,:,2].flatten()[ ~numpy.isnan( self.frame[:,:,2].flatten() ) ]
		else:
			#return self.frame[:,:,2]
			return (numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq - self.time_seq)[1::], (self.shift_pos[1::,:,0] - self.shift_pos[0:-1,:,0])

	def getAllDeltaY ( self, is_flat = False ):
		if is_flat :
			return self.frame[:,:,3].flatten()[ ~numpy.isnan( self.frame[:,:,3].flatten() ) ]
		else:
			#return self.frame[:,:,3]		
			return (numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq - self.time_seq)[1::], (self.shift_pos[1::,:,1] - self.shift_pos[0:-1,:,1])

	def getAllXPath ( self ):
		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq - self.time_seq, self.shift_pos[:,:,0]

	def getAllYPath ( self ):
		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq - self.time_seq, self.shift_pos[:,:,1]

	def getAverageX ( self ) :
		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq, numpy.nanmean( self.shift_pos[:,:,0], axis = 1 ), numpy.nanstd( self.shift_pos[:,:,0], axis = 1 )

	def getAverageY ( self ):
		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq, numpy.nanmean( self.shift_pos[:,:,1], axis = 1 ), numpy.nanstd( self.shift_pos[:,:,1], axis = 1 )		

	def getAllXPSD ( self ):

		time, xpoint = self.getAllXPath()

		time_psd = numpy.full( xpoint.shape, numpy.nan  )
		x_psd = numpy.full( xpoint.shape, numpy.nan )
		for i in range( xpoint.shape[1] ):
			i_nonnan = numpy.where( ~numpy.isnan( xpoint[:,i] ) )[0]
			freq, psd = welch( xpoint[i_nonnan,i], 1/(60*self.time_seq), nperseg = len(i_nonnan) )

			time_psd[ 0:len(freq),i ] = freq
			x_psd[ 0:len(psd),i ] = psd

		return time_psd, x_psd

	def getAverageXPSD ( self, bins = 50 ):
		
		time, psd = self.getAllXPSD()

		xtime = time.flatten()
		xpsd = psd.flatten()
		i_nonnan = numpy.where( ~numpy.isnan(xpsd) )[0]
				
		xtime = xtime[ i_nonnan ]
		xpsd = xpsd[ i_nonnan ]

		ybin, xbin, binnumber = binned_statistic( xtime, xpsd, 'mean', bins = bins )

		return xbin, ybin 

	def getAllYPSD ( self ):

		time, xpoint = self.getAllYPath()

		time_psd = numpy.full( xpoint.shape, numpy.nan  )
		x_psd = numpy.full( xpoint.shape, numpy.nan )
		for i in range( xpoint.shape[1] ):
			i_nonnan = numpy.where( ~numpy.isnan( xpoint[:,i] ) )[0]
			freq, psd = welch( xpoint[i_nonnan,i], 1/(60*self.time_seq), nperseg = len(i_nonnan) )

			time_psd[ 0:len(freq),i ] = freq
			x_psd[ 0:len(psd),i ] = psd

		return time_psd, x_psd

	def getAverageYPSD ( self ):

		time, psd = self.getAllYPSD()

		xtime = time.flatten()
		xpsd = psd.flatten()
		i_nonnan = numpy.where( ~numpy.isnan(xpsd) )[0]
				
		xtime = xtime[ i_nonnan ]
		xpsd = xpsd[ i_nonnan ]

		ybin, xbin, binnumber = binned_statistic( xtime, xpsd, 'mean', bins = bins )

		return xbin, ybin

	def getAverageFMIxByStep ( self ):
		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq, numpy.nanmean( self.shift_pos[:,:,0]/self.shift_pos[:,:,3], axis = 1 ), numpy.nanstd( self.shift_pos[:,:,0]/self.shift_pos[:,:,3], axis = 1 )

	def getAverageFMIyByStep ( self ):
		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq, numpy.nanmean( self.shift_pos[:,:,1]/self.shift_pos[:,:,3], axis = 1 ), numpy.nanstd( self.shift_pos[:,:,1]/self.shift_pos[:,:,3], axis = 1 )

	def getAllVelocityX ( self ):
		return self.frame[:,:,6]

	def getAllVelocityY ( self ):
		return self.frame[:,:,5]

	def getAllVelocity ( self, is_flat = False ):
		if is_flat :
			return self.frame[:,:,7].flatten()[ ~numpy.isnan( self.frame[:,:,7].flatten() ) ]
		else:
			return self.frame[:,:,7]

	def getAllVelocityCorrelation ( self ):
		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq, self.shift_corr[:,:,0]

	def getAverageVelocityCorrelation ( self ):
		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq, numpy.nanmean(self.shift_corr[:,:,0], axis = 1), numpy.nanstd(self.shift_corr[:,:,0], axis = 1)

	def getAverageRealPowerSpectrum ( self ):
		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq, numpy.nanmean(self.shift_corr[:,:,1], axis = 1), numpy.nanstd(self.shift_corr[:,:,1], axis = 1)

	def getAverageImgPowerSpectrum ( self ):
		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq, numpy.nanmean(self.shift_corr[:,:,2], axis = 1), numpy.nanstd(self.shift_corr[:,:,2], axis = 1)

	def getDataPersistenceTime ( self ):

		data = []
		i_nonper = numpy.where( numpy.nansum( self.frame[:,:,12], axis = 0 ) > 0 )[0]
		for cell in i_nonper:
			i_nonnan = numpy.where( ~numpy.isnan( self.frame[:,cell,12] ) )[0]
			for i in range( len(i_nonnan) ):
				index_start = int(self.frame_global[cell,50]) if i == 0 else (i_nonnan[i-1])
				time_per = (i_nonnan[i] - index_start)*self.time_seq
				endtoend_per = math.sqrt( math.pow(self.frame[ i_nonnan[i], cell, 0 ] - self.frame[ index_start, cell, 0 ],2) + math.pow(self.frame[ i_nonnan[i], cell, 1 ] - self.frame[ index_start, cell, 1 ],2) )
				path_per = numpy.sum( numpy.sqrt( numpy.power(self.frame[ (index_start+1):(i_nonnan[i]+1),cell,0] - self.frame[ index_start:i_nonnan[i],cell,0],2) + numpy.power(self.frame[ (index_start+1):(i_nonnan[i]+1),cell,1] - self.frame[ index_start:i_nonnan[i],cell,1],2) ) )

				data.append( [ time_per, endtoend_per, path_per ])

		return numpy.array(data)
		
	def getAllAngleOrientation ( self, is_flat = False ):
		if is_flat :
			return self.frame[:,:,10].flatten()[ ~numpy.isnan( self.frame[:,:,10].flatten() ) ]
		else:
			return self.frame[:,:,10]

	def getFlatAllAngleOrientation ( self ):
		return self.frame[:,:,10].flatten()[ ~numpy.isnan( self.frame[:,:,10].flatten() ) ]

	def getAllDifferenceAngleOrientation( self ):
		return (self.frame[1:self.frame.shape[0]-1,:,10] - self.frame[0:self.frame.shape[0]-2,:,10])

	def getPackingCoefficient ( self, length_window = 5 ):

		cols = numpy.where( numpy.sum( ~numpy.isnan(self.shift_pos[:,:,0]), axis = 0 ) >= (2*length_window) )[0]
		pc = numpy.full( ( self.shift_pos.shape[0], len(cols) ) , numpy.nan )
		for i in cols:
			distance = numpy.power( self.shift_pos[1::,i,0] - self.shift_pos[0:-1:,i,0],2 ) + numpy.power( self.shift_pos[1::,i,1] - self.shift_pos[0:-1:,i,1],2 )
			power_hull = []
			for j in range( self.shift_pos.shape[0] - length_window ):
				if numpy.sum( numpy.isnan( self.shift_pos[j:(j+length_window):,i,0] ) ):
					power_hull.append( math.nan )
				else:
					hull = scipy.spatial.ConvexHull( numpy.column_stack( ( self.shift_pos[j:(j+length_window):,i,0], self.shift_pos[j:(j+length_window):,i,1] ) ) )
					power_hull.append( math.pow(hull.volume,2) )

			rs = numpy.convolve( distance[0:len(power_hull):]/power_hull, numpy.ones(length_window), mode='valid' )
			pc[0:len(rs):,i] = rs

		return numpy.arange(0, pc.shape[0], 1, dtype = int)*self.time_seq, pc, cols

	# MME, MSD & TAMSD

	def getMME ( self ):
		avg_mme = []
		std_mme = []
		for i in range( self.shift_pos.shape[0] ):
			r_max = numpy.nanmax( numpy.power(self.shift_pos[0:i+1,:,0] - self.shift_pos[0,:,0],2) + numpy.power( self.shift_pos[0:i+1,:,1] - self.shift_pos[0,:,1], 2), axis = 0 )
			n_traj = numpy.sum( ~numpy.isnan( self.shift_pos[i,:,0] ) )
			if n_traj <= 0 :
				avg_mme.append( math.nan )
				std_mme.append( math.nan )
			else:
				avg_mme.append( numpy.sum(r_max)/n_traj )
				std_mme.append( numpy.sum( numpy.power( r_max - numpy.sum(r_max)/n_traj,2 ) )/n_traj ) 

		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)[1::]*self.time_seq, numpy.array(avg_mme)[1::], numpy.array(std_mme)[1::]

	def getAllTAMME ( self ):

		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)[1::]*self.time_seq, self.shift_tamme[1::,:]

	def getMSD ( self ):
		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)[1::]*self.time_seq, numpy.nanmean( numpy.power(self.shift_pos[:,:,0] - self.shift_pos[0,:,0],2) + numpy.power( self.shift_pos[:,:,1] - self.shift_pos[0,:,1], 2), axis = 1)[1::], numpy.nanstd( numpy.power(self.shift_pos[:,:,0] - self.shift_pos[0,:,0],2) + numpy.power( self.shift_pos[:,:,1] - self.shift_pos[0,:,1], 2), axis = 1)[1::]

	def getAllTAMSD ( self ):
		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)[1::]*self.time_seq, self.shift_tamsd[1::,:,0]

	def getAllXTAMSD ( self ):
		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq, self.frame[:,:,18]

	def getAllYTAMSD ( self ):
		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq, self.frame[:,:,19]

	def getAverageTAMSD ( self ):
		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)[1::]*self.time_seq, numpy.nanmean(self.shift_tamsd[:,:,0], axis = 1)[1::], numpy.nanstd(self.shift_tamsd[:,:,0], axis = 1)[1::]

	def getAverageTAMME ( self ):
		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)[1::]*self.time_seq, numpy.nanmean(self.shift_tamme, axis = 1)[1::], numpy.nanstd(self.shift_tamme, axis = 1)[1::]
	
	def getAmplitudeTAMSD ( self, is_flat = False, is_flat_not_nan = False ):
		if is_flat :
			return numpy.divide( self.shift_tamsd[:,:,0].T, numpy.nanmean(self.shift_tamsd[:,:,0], axis = 1) ).T[1::,:].flatten()
		elif is_flat_not_nan :
			xdata = numpy.divide( self.shift_tamsd[:,:,0].T, numpy.nanmean(self.shift_tamsd[:,:,0], axis = 1) ).T[1::,:].flatten()
			i_nonnan = numpy.where( ~numpy.isnan(xdata) )[0]
			return xdata[i_nonnan]
		else:
			return numpy.arange(0, self.frame.shape[0], 1, dtype = int)[1::]*self.time_seq, numpy.divide( self.shift_tamsd[:,:,0].T, numpy.nanmean(self.shift_tamsd[:,:,0], axis = 1) ).T[1::,:]

	def getErgodicityBreakingParameterTAMSD ( self ):
		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)[1::]*self.time_seq, ( numpy.nanmean(numpy.power( self.shift_tamsd[1::,:,0],2 ), axis = 1) - numpy.power( numpy.nanmean(self.shift_tamsd[:,:,0], axis = 1)[1::], 2 ) )/numpy.power( numpy.nanmean(self.shift_tamsd[:,:,0], axis = 1)[1::], 2 )
	
	def getGaussianityParameter ( self ):
		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)[1::]*self.time_seq, numpy.nanmean( self.shift_tamsd[1::,:,1], axis = 1)/(numpy.power( numpy.nanmean( self.shift_tamsd[1::,:,0], axis = 1),2 )*2) - 1

	def getRelativeFluctuationsTAMSD ( self ):
		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)[1::]*self.time_seq, numpy.nanmean( numpy.abs( (self.shift_tamsd[1::,:,0].T - numpy.nanmean(self.shift_tamsd[:,:,0], axis = 1)[1::]).T ), axis = 1 )/(numpy.nanmean(self.shift_tamsd[:,:,0], axis = 1)[1::])

	def getTAMSDMSDRatio ( self ):
		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)[1::]*self.time_seq, numpy.nanmean(self.shift_tamsd[:,:,0], axis = 1)[1::]/(numpy.nanmean( numpy.power(self.shift_pos[:,:,0] - self.shift_pos[0,:,0],2) + numpy.power( self.shift_pos[:,:,1] - self.shift_pos[0,:,1], 2), axis = 1)[1::])

	def getMomentRatioMSD ( self ):
		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)[1::]*self.time_seq, (numpy.nanmean( numpy.power( numpy.power(self.shift_pos[:,:,0] - self.shift_pos[0,:,0],2) + numpy.power( self.shift_pos[:,:,1] - self.shift_pos[0,:,1], 2),2 ), axis = 1)[1::])/( numpy.power( numpy.nanmean( numpy.power(self.shift_pos[:,:,0] - self.shift_pos[0,:,0],2) + numpy.power( self.shift_pos[:,:,1] - self.shift_pos[0,:,1], 2), axis = 1)[1::], 2 ) )

	def getMomentsMSD ( self, scaling_exponent = [2] ):

		moments = []
		for exp in scaling_exponent:
			moments.append( numpy.nanmean( numpy.power( numpy.sqrt( numpy.power(self.shift_pos[:,:,0] - self.shift_pos[0,:,0],2) + numpy.power( self.shift_pos[:,:,1] - self.shift_pos[0,:,1], 2) ), exp), axis = 1)[1::] )
		
		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)[1::]*self.time_seq, moments
		
	def getMomentRatioMME ( self ):
		avg_mme = []
		avg_mme_4 = []
		for i in range( self.shift_pos.shape[0] ):
			r_max = numpy.nanmax( numpy.power(self.shift_pos[0:i+1,:,0] - self.shift_pos[0,:,0],2) + numpy.power( self.shift_pos[0:i+1,:,1] - self.shift_pos[0,:,1], 2), axis = 0 )
			n_traj = numpy.sum( ~numpy.isnan( self.shift_pos[i,:,0] ) )
			if n_traj <= 0 :
				avg_mme.append( math.nan )
				avg_mme_4.append( math.nan )
			else:
				avg_mme.append( numpy.sum(r_max)/n_traj )
				avg_mme_4.append( numpy.sum( numpy.power(r_max,2) )/n_traj )
		
		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)[1::]*self.time_seq, numpy.array(avg_mme_4)[1::]/(numpy.power(avg_mme,2)[1::])

	def getXAverageMixingBreakingDynamicalFunctionalTest ( self ):

		D = numpy.nanmean( numpy.exp( 1j*( self.shift_pos[:,:,0] - self.shift_pos[0,:,0] ) ), axis = 1 )
		a = numpy.power( numpy.abs( numpy.nanmean( numpy.exp( 1j*self.shift_pos[0,:,0] ) ) ), 2)

		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq, (D-a), numpy.cumsum( D - a )/(numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq)

	def getYAverageMixingBreakingDynamicalFunctionalTest ( self ):

		D = numpy.nanmean( numpy.exp( 1j*(self.shift_pos[:,:,1] - self.shift_pos[0,:,1]) ), axis = 1 )
		a = numpy.power( numpy.abs( numpy.nanmean( numpy.exp( 1j*self.shift_pos[0,:,1] ) ) ), 2)

		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq, (D-a), numpy.cumsum( D - a )/(numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq)

	def getXAllMixingBreakingDynamicalFunctionalTest ( self ):

		mixingergo = numpy.full( ( self.shift_pos.shape[0], self.shift_pos.shape[1] ), numpy.nan )

		for i in range( self.shift_pos.shape[1] ):
			i_nonnan = numpy.where( ~numpy.isnan(self.shift_pos[:,i,0]) )[0]
			a = 0
			for j in i_nonnan:
				a = a + numpy.exp( 1j*self.shift_pos[j,i,0] )
			a = numpy.power( numpy.abs(a/len(i_nonnan)), 2)

			for j in range( len(i_nonnan) ):
				D = 0
				for k in range( 0, len(i_nonnan) - j ):
					D = D + numpy.exp( 1j*(self.shift_pos[i_nonnan[k+j],i,0] - self.shift_pos[i_nonnan[k],i,0]) )
				mixingergo[i_nonnan[j],i] = (D/(len(i_nonnan) - j)) - a

		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq, mixingergo, ((numpy.cumsum( mixingergo, axis = 0 ).T)/ (numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq)).T

	def getYAllMixingBreakingDynamicalFunctionalTest ( self ):

		mixingergo = numpy.full( ( self.shift_pos.shape[0], self.shift_pos.shape[1] ), numpy.nan )

		for i in range( self.shift_pos.shape[1] ):
			i_nonnan = numpy.where( ~numpy.isnan(self.shift_pos[:,i,0]) )[0]
			a = 0
			for j in i_nonnan:
				a = a + numpy.exp( 1j*self.shift_pos[j,i,1]  )
			a = numpy.power( numpy.abs(a/len(i_nonnan)), 2)

			for j in range( len(i_nonnan) ):
				D = 0
				for k in range( 0, len(i_nonnan) - j ):
					D = D + numpy.exp( 1j*(self.shift_pos[i_nonnan[k+j],i,1] - self.shift_pos[i_nonnan[k],i,1]) )
				mixingergo[i_nonnan[j],i] = (D/(len(i_nonnan) - j)) - a

		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq, mixingergo, ((numpy.cumsum( mixingergo, axis = 0 ).T)/ (numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq)).T

	# getter ageing

	def getAllTAMSDWithAgeing ( self ):
		all_tamsd = self.frame[:,:,8]
		all_tamsd[self.frame_global[:,40].astype('int'),range(self.frame_global.shape[0])] = numpy.nan
		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq, self.frame[:,:,8]

	def getAverageTAMSDWithAgeing ( self ):
		all_tamsd = self.frame[:,:,8]
		all_tamsd[self.frame_global[:,40].astype('int'),range(self.frame_global.shape[0])] = numpy.nan
		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq, numpy.nanmean( all_tamsd, axis = 1 ), numpy.nanstd( all_tamsd, axis = 1 )

	def getMSDWithAgeing ( self ):
		all_squared = numpy.power( self.frame[:,:,0] - self.frame[self.frame_global[:,40].astype('int'),range(self.frame_global.shape[0]),0] , 2) + numpy.power( self.frame[:,:,1] - self.frame[self.frame_global[:,40].astype('int'),range(self.frame_global.shape[0]),1] , 2)
		all_squared[self.frame_global[:,40].astype('int'),range(self.frame_global.shape[0])] = numpy.nan
		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq, numpy.nanmean( all_squared , axis = 1), numpy.nanstd( all_squared , axis = 1)

	def getAmplitudeTAMSDWithAging ( self ):
		all_tamsd = self.frame[:,:,8]
		all_tamsd[self.frame_global[:,40].astype('int'),range(self.frame_global.shape[0])] = numpy.nan
		if is_flat :
			return numpy.divide( all_tamsd.T, numpy.nanmean(all_tamsd, axis = 1) ).T.flatten()
		elif is_flat_not_nan :
			xdata = numpy.divide( all_tamsd.T, numpy.nanmean(all_tamsd, axis = 1) ).T.flatten()
			i_nonnan = numpy.where( ~numpy.isnan(xdata) )[0]
			return xdata[i_nonnan]			
		else:
			return numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq, numpy.divide( all_tamsd.T, numpy.nanmean(all_tamsd, axis = 1) ).T
	
	def getErgodicityBreakingParameterTAMSDWithAgeing ( self ):
		all_tamsd = self.frame[:,:,8]
		all_tamsd[self.frame_global[:,40].astype('int'),range(self.frame_global.shape[0])] = numpy.nan

		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq, 
		( numpy.nanmean(numpy.power( all_tamsd,2 ), axis = 1) - numpy.power( numpy.nanmean(all_tamsd, axis = 1), 2 ) )/numpy.power( numpy.nanmean(all_tamsd, axis = 1), 2 )

	def getTAMSDMSDRatioWithAgeing ( self ):
		all_squared = numpy.power( self.frame[:,:,0] - self.frame[self.frame_global[:,40].astype('int'),range(self.frame_global.shape[0]),0] , 2) + numpy.power( self.frame[:,:,1] - self.frame[self.frame_global[:,40].astype('int'),range(self.frame_global.shape[0]),1] , 2)
		all_squared[self.frame_global[:,40].astype('int'),range(self.frame_global.shape[0])] = numpy.nan
		all_tamsd = self.frame[:,:,8]
		all_tamsd[self.frame_global[:,40].astype('int'),range(self.frame_global.shape[0])] = numpy.nan

		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq, numpy.nanmean(all_tamsd, axis = 1)/numpy.nanmean( all_squared, axis = 1)
	
	# getter details Persistence, FMI, RatioXY

	def getAllPersistenceDistance ( self ):
		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq, self.shift_persistence_distance[:,:,0]

	def getAveragePersistenceDistance ( self ):
		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq, numpy.nanmean(self.shift_persistence_distance[:,:,0], axis = 1), numpy.nanstd(self.shift_persistence_distance[:,:,0], axis = 1)

	def getAllFMIx ( self ):
		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq, self.shift_persistence_distance[:,:,1]

	def getAllFMIy ( self ):
		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq, self.shift_persistence_distance[:,:,2]

	def getAverageFMIx ( self ):
		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq, numpy.nanmean(self.shift_persistence_distance[:,:,1], axis = 1), numpy.nanstd(self.shift_persistence_distance[:,:,1], axis = 1)

	def getAverageFMIy ( self ):
		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq, numpy.nanmean(self.shift_persistence_distance[:,:,2], axis = 1), numpy.nanstd(self.shift_persistence_distance[:,:,2], axis = 1)
	
	def getAverageXYRatio ( self ):
		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq, numpy.nanmean( numpy.absolute(self.shift_persistence_distance[:,:,1]), axis = 1)/numpy.nanmean( numpy.absolute(self.shift_persistence_distance[:,:,2]), axis = 1) 

	def getAverageAngleTurning ( self ):
		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq, self.shift_angle[:,:,0]

	# getters Characteristic

	def getDivisionTime ( self ):
		return self.divison_time

	def getCellSize ( self ):
		return self.cell_size

	def getAspectRatio ( self ):
		return numpy.array(self.aspect_ratio)[ ~numpy.isnan(self.aspect_ratio) ].tolist()

	def getGrowthRate ( self ):
		return self.growth_rate

	# get Global Speed

	def getGSpeed ( self ):
		return self.g_speed
	def getStdGSpeed ( self ):
		return self.std_g_speed

	# getters GLOBAL

	# blob diameter
	def getGlobalDiameterBlob ( self ):
		return self.frame_global[:,2]

	# maximum distance traveled
	def getGlobalMaxDistance ( self ):
		return self.frame_global[:,3]

	# total distance traveled
	def getGlobalPath ( self ):
		return self.frame_global[:,4]

	# net distance traveled
	def getGlobalEndtoEndDistance ( self ):
		return self.frame_global[:,7]

	# x net distance traveled
	def getGlobalXDistance ( self ):
		return self.frame_global[:,5]

	# y net distance travaled
	def getGlobalYDistance ( self ):
		return self.frame_global[:,6]

	# confinement ratio or persistence ratio
	def getGlobalPersistenceRatio ( self ):
		return self.frame_global[:,8]

	# directionality ratio
	def getGlobalDirectionalityRatio ( self, is_not_nan = False ):
		if is_not_nan :
			i_nonnan = numpy.where( ~numpy.isnan(self.frame_global[:,9]) )[0]
			return self.frame_global[i_nonnan,9]
		else:
			return self.frame_global[:,9]

	# displacement ratio
	def getGlobalDisplacementRatio ( self ):
		return self.frame_global[:,18]

	# outreach ratio
	def getGlobalOutreachRatio ( self ):
		return self.frame_global[:,19]

	# explorer ratio
	def getGlobalExplorerRatio ( self ):
		return self.frame_global[:,20]

	# x FMI
	def getGlobalXFMI ( self ):
		return self.frame_global[:,10]

	# y FMI
	def getGlobalYFMI ( self ):
		return self.frame_global[:,11]

	# mean straight-line speed
	def getGlobalMeanStraightLineSpeed ( self ):
		return self.frame_global[:,12]

	# x mean straight line speed
	def getGlobalXMeanStraightLineSpeed ( self ):
		return self.frame_global[:,13]

	# y mean straight line speed
	def getGlobalYMeanStraightLineSpeed ( self ):
		return self.frame_global[:,14]
	
	# total speed
	def getGlobalTotalSpeed ( self ):
		return self.frame_global[:,15]

	# mean curvilinear speed
	def getGlobalMeanCurvilinearSpeed ( self ):
		return self.frame_global[:,16]

	# linearity of forward progression
	def getGlobalLinearityForwardProgression ( self ):
		return self.frame_global[:,17]

	# hull perimeter
	def getGlobalHullPerimeter ( self ):
		return self.frame_global[:,21]

	# hull area
	def getGlobalHullArea ( self ):
		return self.frame_global[:,22]

	# acircularity
	def getGlobalAcircularity ( self ):
		return self.frame_global[:,23]

	# mass center orientation
	def getGlobalMassCenterOrientation ( self ):
		i_nonnan = numpy.where( ~numpy.isnan(self.frame_global[:,24]) )[0]
		return self.frame_global[i_nonnan,24]

	# first orientation cell body
	def getGlobalFirstOrientationCellBody ( self ):
		i_nonnan = numpy.where( ~numpy.isnan(self.frame_global[:,25]) )[0]
		return self.frame_global[i_nonnan,25]

	# max view angle 
	def getGlobalMaxViewAngle ( self ):
		i_nonnan = numpy.where( ~numpy.isnan(self.frame_global[:,26]) )[0]
		return self.frame_global[i_nonnan,26]

	# tracking view angle 
	def getGlobalTrackingViewAngle ( self ):
		i_nonnan = numpy.where( ~numpy.isnan(self.frame_global[:,27]) )[0]
		return self.frame_global[i_nonnan,27]

	# get scaling exponent fit
	def getScalingExponentFit ( self ):
		return self.frame_global[:,28]

	# get generalised diffusion coeficient
	def getGeneralisedDiffusionCoefficientFit ( self ):
		return self.frame_global[:,29]

	# get first scaling exponent fit
	def getFirstScalingExponentFit ( self ):
		return self.frame_global[:,37]

	# get first generalised diffusion coeficient
	def getFirstGeneralisedDiffusionCoefficientFit ( self ):
		return self.frame_global[:,38]

	# get X scaling exponent and generalised diffusion coeficient
	def getXAllScalingExponentAndGeneraisedDiffusionCoefficient ( self, window = (0,10) ):
		def fit_tamsd(t, alpha, org):
			return alpha*t + org
				
		xtime, xtamsd = self.getAllXTAMSD()
		xexponent = []
		xcoefficient = []
		for i in range( xtamsd.shape[1] ):
			
			split_time = xtime[ int(self.frame_global[i,50])+1+window[0]:int(self.frame_global[i,50])+1+window[1] ] - xtime[ int(self.frame_global[i,50]) ]
			split_average = xtamsd[ int(self.frame_global[i,50])+1+window[0]:int(self.frame_global[i,50])+1+window[1],i ]

			xrest = curve_fit( fit_tamsd, numpy.log10(split_time), numpy.log10(split_average) )
			
			xexponent.append( xrest[0][0] )
			xcoefficient.append( math.pow(10, xrest[0][1]) )

		return numpy.array(xexponent), numpy.array(xcoefficient)

	# get Y scaling exponent and generalised diffusion coeficient
	def getYAllScalingExponentAndGeneraisedDiffusionCoefficient ( self, window = (0,10) ):
		def fit_tamsd(t, alpha, org):
			return alpha*t + org
				
		xtime, xtamsd = self.getAllYTAMSD()
		xexponent = []
		xcoefficient = []
		for i in range( xtamsd.shape[1] ):
			
			split_time = xtime[ int(self.frame_global[i,50])+1+window[0]:int(self.frame_global[i,50])+1+window[1] ] - xtime[ int(self.frame_global[i,50]) ]
			split_average = xtamsd[ int(self.frame_global[i,50])+1+window[0]:int(self.frame_global[i,50])+1+window[1],i ]

			xrest = curve_fit( fit_tamsd, numpy.log10(split_time), numpy.log10(split_average) )
			
			xexponent.append( xrest[0][0] )
			xcoefficient.append( math.pow(10, xrest[0][1]) )

		return numpy.array(xexponent), numpy.array(xcoefficient)

	# get start time
	def getGlobalStartTime ( self ):
		return self.frame_global[:,40]

	# get end time
	def getGlobalEndTime ( self ):
		return self.frame_global[:,41]

	# get start index
	def getGlobalStartIndex ( self ):
		return self.frame_global[:,50].astype('int')

	# get end index
	def getGlobalEndIndex ( self ):
		return self.frame_global[:,51].astype('int')

	# gyration tensor
	def getGyrationTensor ( self ):
		return self.frame_global[:,30:33]

	# gyration angle
	def getGyrationAngle ( self ):
		return self.frame_global[:,33]

	# gyration eigenvalues
	def getGyrationEigenValues ( self ):
		return numpy.stack( (self.frame_global[:,34], self.frame_global[:,35]), axis = 1 )

	# gyration radius
	def getGyrationRadius ( self ):
		return (self.frame_global[:,34] + self.frame_global[:,35])

	# gyration asymmetry ratio a2
	def getGyrationAsymmetryRatio_a2 ( self ):
		return self.frame_global[:,35]/self.frame_global[:,34]

	# gyration asymmetry ratio A2
	def getGyrationAsymmetryRatio_A2 ( self ):
		return numpy.power(self.frame_global[:,34] - self.frame_global[:,35],2)/numpy.power( self.frame_global[:,34] + self.frame_global[:,35],2)

	# gyration asymmetry A
	def getGyrationAsummetry_A ( self ):
		return -numpy.log( 1 - numpy.power(self.frame_global[:,34] - self.frame_global[:,35],2)/(2*numpy.power( self.frame_global[:,34] + self.frame_global[:,35],2)) )

	# fractal dimension
	def getGlobalFractalDimension ( self ):
		return self.frame_global[:,36]

	# length points
	def getGlobalLengthPoints ( self ):
		return self.frame_global[:,39]

	# start cell aspect ratio
	def getGlobalStartAspectRatio ( self ):
		return self.frame_global[:,42]/self.frame_global[:,43]

	# end cell aspect ratio
	def getGlobalEndAspectRatio ( self ):
		return self.frame_global[:,44]/self.frame_global[:,45]

	# get TAMME scaling exponent fit
	def getScalingExponentFitTAMME ( self ):
		return self.frame_global[:,46]

	# get TAMME generalised diffusion coeficient
	def getGeneralisedDiffusionCoefficientFitTAMME ( self ):
		return self.frame_global[:,47]

	# get TAMME first scaling exponent fit
	def getFirstScalingExponentFitTAMME ( self ):
		return self.frame_global[:,48]

	# get TAMME first generalised diffusion coeficient
	def getFirstGeneralisedDiffusionCoefficientFitTAMME ( self ):
		return self.frame_global[:,49]	


	def __refreshGraph ( self ):

		self.graph.global_mass_center_orientation = self.getGlobalMassCenterOrientation()
		self.graph.global_first_orientation_cell_body = self.getGlobalFirstOrientationCellBody()
		self.graph.path_x_times, self.graph.path_x_tracks = self.getAllXPath()
		self.graph.path_y_times, self.graph.path_y_tracks = self.getAllYPath()
		self.graph.average_angle_turning_time, self.graph.average_angle_turning = self.getAverageAngleTurning()

	class resource_graph :

		def __init__ ( self ):
			self.global_mass_center_orientation = None
			self.global_first_orientation_cell_body = None
			self.path_x_tracks = None
			self.path_x_times = None
			self.path_y_tracks = None
			self.path_y_times = None

			self.average_angle_turning_time = None
			self.average_angle_turning = None

		def pathXTracks ( self, is_sub = False, color = 'black' ) :

			if not is_sub :
				plt.figure( figsize = (6,4) )

			plt.plot( self.path_x_times, self.path_x_tracks, color = color )
			plt.xlabel('t (min)',fontdict = {'size' : 16})
			plt.ylabel(r'$x \; (\mu m)$', fontdict = {'size' : 16})
			plt.xticks( fontsize = 16 )
			plt.yticks( fontsize = 16 )
			plt.grid( linestyle = ':' )
			return plt

		def pathYTracks ( self, is_sub = False, color = 'black' ) :
			
			if not is_sub :
				plt.figure( figsize = (6,4) )

			plt.plot( self.path_y_times, self.path_y_tracks, color = color )
			plt.xlabel('t (min)',fontdict = {'size' : 16})
			plt.ylabel(r'$y \; (\mu m)$', fontdict = {'size' : 16})
			plt.xticks( fontsize = 16 )
			plt.yticks( fontsize = 16 )
			plt.grid( linestyle = ':' )
			return plt

		def pathXYTracks ( self, is_sub = False, color = 'black' ) :

			if not is_sub :
				plt.figure( figsize = (6,4) )

			plt.plot( self.path_x_tracks, self.path_y_tracks, color = color )
			plt.xlabel(r'$x \; (\mu m)$',fontdict = {'size' : 16})
			plt.ylabel(r'$y \; (\mu m)$', fontdict = {'size' : 16})
			plt.axis( 'equal' )
			plt.xticks( fontsize = 16 )
			plt.yticks( fontsize = 16 )
			plt.grid( linestyle = ':' )

			return plt

		def globalMassCenterOrientation ( self, is_sub = True, rows = 1, cols = 1, index = 1, title = '', color = '.8', bin_size = 20 ) :

			if is_sub :
				plt.subplot(rows, cols, index, projection = 'polar')

			plt.title(title+' \n', fontdict = { 'fontsize' : 16 })

			a , b = numpy.histogram( self.global_mass_center_orientation , bins=numpy.arange(0, 360+bin_size, bin_size))
			centers = numpy.deg2rad(numpy.ediff1d(b)//2 + b[:-1])

			plt.bar(centers, a, width = numpy.deg2rad(bin_size), bottom=0.0, color=color , edgecolor='k')

			return plt
		
		def globalFirstOrientationCellBody ( self, is_sub = True, rows = 1, cols = 1, index = 1, title = '', color = '.8', bin_size = 20 ) :

			if is_sub :
				plt.subplot(rows, cols, index, projection = 'polar')

			plt.title(title+' \n', fontdict = { 'fontsize' : 16 })

			a , b = numpy.histogram( self.global_first_orientation_cell_body , bins=numpy.arange(0, 360+bin_size, bin_size))
			centers = numpy.deg2rad(numpy.ediff1d(b)//2 + b[:-1])

			plt.bar(centers, a, width = numpy.deg2rad(bin_size), bottom=0.0, color=color, edgecolor='k')

			return plt

		def AverageAngleTurning ( self, is_sub = False ):

			if not is_sub :
				plt.figure( figsize = (6,4) ) 

			plt.plot( self.average_angle_turning_time, self.average_angle_turning )
			plt.xlabel('t (min)',fontdict = {'size' : 16})
			plt.ylabel(r'$ \theta $', fontdict = {'size' : 16})
			plt.xticks( fontsize = 16 )
			plt.yticks( fontsize = 16 )
			plt.grid( linestyle = ':' )


