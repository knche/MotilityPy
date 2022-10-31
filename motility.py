import numpy
import math
import smallestenclosingcircle
import scipy.spatial
import matplotlib.pyplot as plt

class motility :

	def __init__ ( self, data= [], micron_px = 1, time_seq = 1, time_acquisition = 0 ):
		self.data = data
		self.micron_px = micron_px
		self.time_seq = time_seq
		self.time_adquisition = time_acquisition
		self.frame = numpy.full( ( time_acquisition, len(self.data), 20 ), numpy.nan )
		self.divison_time = []
		self.cell_size = []
		self.aspect_ratio = []
		self.growth_rate = []
		self.shift_pos = numpy.full( ( self.frame.shape[0], self.frame.shape[1], 4 ), numpy.nan )
		self.shift_corr = numpy.full( ( self.frame.shape[0], self.frame.shape[1], 3 ), numpy.nan )
		self.shift_msd = numpy.full( (self.frame.shape[0], self.frame.shape[1] ), numpy.nan )
		self.shift_persistence_distance = numpy.full( (self.frame.shape[0], self.frame.shape[1], 3), numpy.nan )
		self.shift_angle = numpy.full( (self.frame.shape[0], self.frame.shape[1], 2 ), numpy.nan )
		self.persistence_time = []
		self.persistence_speed = []
		self.frame_global = numpy.full( ( len(self.data), 30 ), numpy.nan )

		self.g_speed = 0
		self.std_g_speed = 0

		self.graph = self.resource_graph()

	def run ( self ):

		# x, y, angle
		for i in range( len(self.data) ):
			for j in range( len(self.data[i]) ):
				self.frame[self.data[i][j][0],i,0] = self.data[i][j][3]*self.micron_px
				self.frame[self.data[i][j][0],i,1] = self.data[i][j][4]*self.micron_px
				self.frame[self.data[i][j][0],i,10] = self.data[i][j][5]*self.micron_px

				if j > 0 :
					self.frame[self.data[i][j-1][0],i,15] = math.atan( (self.data[i][j][4] - self.data[i][0][4])/(self.data[i][j][3] - self.data[i][0][3]) )

		# deltaX, deltaY, distance, velocity x, velocity x, instant velocity
		self.frame[1:self.time_adquisition,:,2] = self.frame[1:self.time_adquisition,:,0] - self.frame[0:self.time_adquisition-1,:,0]
		self.frame[1:self.time_adquisition,:,3] = self.frame[1:self.time_adquisition,:,1] - self.frame[0:self.time_adquisition-1,:,1]
		self.frame[:,:,4] = numpy.sqrt( numpy.power( self.frame[:,:,2], 2) + numpy.power( self.frame[:,:,3], 2) )
		self.frame[:,:,5] = self.frame[:,:,2]/self.time_seq
		self.frame[:,:,6] = self.frame[:,:,3]/self.time_seq
		self.frame[:,:,7] = self.frame[:,:,4]/self.time_seq

		# msd
		for i in range( len(self.data) ):
			for j in range( len(self.data[i]) ):
				sum_msd = 0
				for k in range( 0, len(self.data[i]) - j ):
					sum_msd = sum_msd + math.pow(self.data[i][k+j][3]*self.micron_px - self.data[i][k][3]*self.micron_px,2) + math.pow(self.data[i][k+j][4]*self.micron_px - self.data[i][k][4]*self.micron_px,2)
				self.frame[self.data[i][j][0],i,8] = (sum_msd)/( len(self.data[i]) - j )

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
		per25_t_div = numpy.percentile(self.divison_time, 25)/self.time_seq
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
						self.frame[xinit_position,i,12] = 1

		# run rollXYDistance
		self.rollXYDistance()
		self.rollVelocityCorrelation()
		self.computePersistenceTime()
		self.rollMSD()
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

		self.__refreshGraph()


	# dtype = path_length, blob, end_to_end_distance, max_distance, time
	# operation = range, equal, greater, smaller, interquartile1.5x, 1.5xonlythirdquartile, interquartile, notequaltoacquisition
	# ft = percentile, bias
	def filter ( self, dtype = 'path_length', ft = 'percentile', operation = 'range', start_value = 25, end_value = 0, offset_time = 10 ):

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

		self.frame = self.frame[:,index,:]
		self.shift_pos = self.shift_pos[:,index,:]
		self.shift_corr = self.shift_corr[:,index,:]
		self.shift_msd = self.shift_msd[:,index]
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

	def rollMSD ( self ):

		for i in range(self.frame.shape[1]):
			xlen = len(self.frame[~numpy.isnan(self.frame[:,i,8]),i,8])
			self.shift_msd[0:xlen,i] = self.frame[~numpy.isnan(self.frame[:,i,8]),i,8]

	def rollPersistenceDistance ( self ):

		for i in range(self.frame.shape[1]):
			xlen = len(self.frame[~numpy.isnan(self.frame[:,i,9]),i,9])
			self.shift_persistence_distance[0:xlen,i,0] = self.frame[~numpy.isnan(self.frame[:,i,9]),i,9]
			self.shift_persistence_distance[0:xlen,i,1] = self.frame[~numpy.isnan(self.frame[:,i,13]),i,13]
			self.shift_persistence_distance[0:xlen,i,2] = self.frame[~numpy.isnan(self.frame[:,i,14]),i,14]

	def computePersistenceTime( self ):

		for i in range( self.frame.shape[1] ):
			i_nonnan = numpy.where( ~numpy.isnan(self.frame[:,i,12]) )[0]
			if len(i_nonnan) > 0:
				for j in range( 1,len(i_nonnan) ):
					self.persistence_time.append( i_nonnan[j] - i_nonnan[j-1] )
					self.persistence_speed.append( numpy.nanmean( numpy.abs( self.frame[i_nonnan[j-1]:i_nonnan[j]+1,i,7] ) ) )

	def rollAngle ( self ) :

		for i in range( self.frame.shape[1] ):
			i_nonnan = numpy.where( ~numpy.isnan(self.frame[:,i,15]) )[0]
			self.shift_angle[:,i,0] = numpy.roll(self.frame[:,i,15], self.frame.shape[0] - i_nonnan[0] )


	def computeGSpeed ( self, explore_fit = 0 ):
		if type( explore_fit ) is list:
			xgspeed = []
			for fit in explore_fit:
				xgspeed.append( ( fit-1/numpy.average(self.divison_time) )*numpy.average(self.cell_size) )
			self.g_speed = numpy.average( xgspeed )
			self.std_g_speed = numpy.std( xgspeed )
		else:
			self.g_speed = ( explore_fit-1/numpy.average(self.divison_time) )*numpy.average(self.cell_size)

	# getters Details

	def getAllXPath ( self ):
		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq - self.time_seq, self.shift_pos[:,:,0]

	def getAllYPath ( self ):
		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq - self.time_seq, self.shift_pos[:,:,1]

	def getAverageX ( self ) :
		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq, numpy.nanmean( self.shift_pos[:,:,0], axis = 1 ), numpy.nanstd( self.shift_pos[:,:,0], axis = 1 )

	def getAverageY ( self ):
		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq, numpy.nanmean( self.shift_pos[:,:,1], axis = 1 ), numpy.nanstd( self.shift_pos[:,:,1], axis = 1 )		

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
		return self.persistence_speed, numpy.array(self.persistence_time)*self.time_seq

	def getModePersistenceTime ( self ):
		val_s, count_s = numpy.unique( self.persistence_speed, return_counts = True )
		val_t, count_t = numpy.unique( self.persistence_time, return_counts = True )

		return numpy.average(val_s[ numpy.argwhere( count_s == numpy.max(count_s) ) ]), numpy.average(val_t[ numpy.argwhere( count_t == numpy.max(count_t) ) ])*self.time_seq

	def getMeanPersistenceTime ( self ):
		return numpy.average(self.persistence_speed), numpy.std(self.persistence_speed), numpy.average(self.persistence_time)*self.time_seq, numpy.std(self.persistence_time)*self.time_seq

	def getMedianPersistenceTime ( self ):
		return numpy.median(self.persistence_speed), numpy.median(self.persistence_time)*self.time_seq

	def getAllAngleOrientation ( self ):
		return self.frame[:,:,10]

	def getFlatAllAngleOrientation ( self ):
		return self.frame[:,:,10].flatten()[ ~numpy.isnan( self.frame[:,:,10].flatten() ) ]

	def getAllDifferenceAngleOrientation( self ):
		return (self.frame[1:self.frame.shape[0]-1,:,10] - self.frame[0:self.frame.shape[0]-2,:,10])

	def getAllMSD ( self ):
		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq, self.shift_msd

	def getAverageMSD ( self ):
		return numpy.arange(0, self.frame.shape[0], 1, dtype = int)*self.time_seq, numpy.nanmean(self.shift_msd, axis = 1), numpy.nanstd(self.shift_msd, axis = 1)

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
	def getGlobalDirectionalityRatio ( self ):
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


