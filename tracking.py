import trackpy
import numpy
import matplotlib.pyplot as plt
import math
import skimage.measure
import skimage.io
import skimage.util
import pandas
import scipy.signal
import smallestenclosingcircle
from scipy.optimize import curve_fit

class tracking :

	def __init__ ( self, files = [], stack_images = 0, min_area = 9, max_length = 0, micron_px = 0.1, time_seq = 1, crop_shift = 0, init_x_img = 0, init_y_img = 0, tolerance_tracking = 90, adaptive_stop_tracking = 2, adaptive_step_tracking = 0.5, memory_tracking = 3, diameter_tracking = 3, is_enabled_half_path_detection = True  ) :

		self.files = files
		self.stack_images = stack_images
		self.min_area =  min_area
		self.max_length = max_length
		self.micron_px = micron_px
		self.time_seq = time_seq
		self.crop_shift = crop_shift
		self.init_x_img = init_x_img
		self.init_y_img = init_y_img
		self.tolerance_tracking = tolerance_tracking
		self.adaptive_stop_tracking = adaptive_stop_tracking
		self.adaptive_step_tracking = adaptive_step_tracking
		self.memory_tracking = memory_tracking
		self.diameter_tracking = diameter_tracking
		self.is_enabled_half_path_detection = is_enabled_half_path_detection
		self.max_images_explorer = 0
		self._is_keep_multiple_explorer = False
		self._is_run = False
		self._is_execute_explore = False

		self.stack_seg = numpy.full( len(self.files), numpy.nan ).tolist()
		self.area = numpy.full( len(self.files), numpy.nan ).tolist()
		self.explore = numpy.full( len(self.files), numpy.nan ).tolist()
		self.regions_seg = numpy.full( len(self.files), numpy.nan ).tolist()
		self.cell_tracking = numpy.full( len(self.files), numpy.nan ).tolist()
		self.find_cells = numpy.full( len(self.files), numpy.nan ).tolist()
		self.only_bacteria_enabled = numpy.full( len(self.files), numpy.nan ).tolist()
		self.find_bacterium = numpy.full( len(self.files), numpy.nan ).tolist()
		self.path_details = numpy.full( len(self.files), numpy.nan ).tolist()
		self.path_global = numpy.full( len(self.files), numpy.nan ).tolist()
		self.length_cells = []

		for i in range( len(self.files) ):
			self.stack_seg[i] = []
			self.area[i] = []
			self.explore[i] = []
			self.regions_seg[i] = []
			self.cell_tracking[i] = []
			self.find_cells[i] = []
			self.only_bacteria_enabled[i] = []
			self.find_bacterium[i] = []
			self.path_details[i] = []
			self.path_global[i] = []


		self.mean_length_cell = 0
		self.std_length_cell = 0

		self.util = self.resource()

	def run ( self ):
		
		if not self._is_execute_explore:
			self.load()
			self.processExplore()

			area, explore = self.getExplore()
			self.util.setTimeArea( area )
			self.util.setTimeExplorer( explore )
			self.util.is_multiple = self._is_keep_multiple_explorer
		
		self.regionsExtract()
		self.avgLengthCells()
		self.trackMaking()
		self.cellsFinding()
		self.processPath()		

		self._is_run = True

	def executeExplorer ( self ):
		
		if not self._is_run :
			self.load()
			self.processExplore()

			area, explore = self.getExplore()
			self.util.setTimeArea( area )
			self.util.setTimeExplorer( explore )
			self.util.is_multiple = self._is_keep_multiple_explorer

		self._is_execute_explore = True


	def load ( self ):
		qty_images_stack = []
		max_length = self.max_length
		if type(self.max_length) is int:
			max_length = [max_length]*len(self.files)

		for i in range( len(self.files) ):
			self.stack_seg[i] = skimage.io.imread( self.files[i] )
			
			width_img = self.stack_seg[i].shape[2] - self.crop_shift
			height_img = self.stack_seg[i].shape[1] - self.crop_shift

			if max_length[i] <= 0 :
				self.stack_seg[i] = self.stack_seg[i][:,self.init_y_img:height_img,self.init_x_img:width_img]
			else:
				self.stack_seg[i] = self.stack_seg[i][0:max_length[i],self.init_y_img:height_img,self.init_x_img:width_img]

			qty_images_stack.append( self.stack_seg[i].shape[0] )

		if self.max_images_explorer <= 0:
			self.max_images_explorer = numpy.nanmin( qty_images_stack )

	def regionsExtract ( self ):

		for j in range( len(self.stack_seg) ):
			for i in range(self.stack_seg[j].shape[0]):
				regions = skimage.measure.regionprops_table( skimage.measure.label(self.stack_seg[j][i,:,:],connectivity=2), properties=['centroid','area','coords','image','eccentricity','major_axis_length','minor_axis_length','orientation'] )
				regions = pandas.DataFrame( regions )
				index_f = []
				for index in regions.index:
					if regions.at[index,'area'] < self.min_area:
						index_f.append( index )
					else:
						self.length_cells.append( regions.at[index,'major_axis_length'] )

				regions = regions.drop( index_f )
				self.regions_seg[j].append( regions )

	def avgLengthCells ( self ):
		self.mean_length_cell = numpy.mean( self.length_cells )
		self.std_length_cell = math.sqrt( numpy.var( self.length_cells ) )

	def trackMaking ( self ):
		for i in range( len(self.stack_seg) ):
			f = pandas.DataFrame( columns = ['y','x','frame','index_frame'] )
			ximages = self.stack_seg[i].shape[0]
			for k in range(ximages):
				for index in self.regions_seg[i][k].index:
					#f = f.append( {'y':self.regions_seg[i][k].at[index,'centroid-0'],'x':self.regions_seg[i][k].at[index,'centroid-1'],'frame':k,'index_frame':index, 'major_axis_length':self.regions_seg[i][k].at[index,'major_axis_length'], 'minor_axis_length':self.regions_seg[i][k].at[index,'minor_axis_length'] }, ignore_index = True )
					xcoords = self.regions_seg[i][k].at[index,'coords']
					xcoords[:,0] = self.stack_seg[i].shape[1] - xcoords[:,0]
					xorientation = (math.pi/2 + self.regions_seg[i][k].at[index,'orientation']) if self.regions_seg[i][k].at[index,'orientation'] < 0 else (self.regions_seg[i][k].at[index,'orientation'] - math.pi/2)
					f = f.append( {'y': self.stack_seg[i].shape[1] - self.regions_seg[i][k].at[index,'centroid-0'],'x':self.regions_seg[i][k].at[index,'centroid-1'],'frame':k,'index_frame':index, 'major_axis_length':self.regions_seg[i][k].at[index,'major_axis_length'], 'minor_axis_length':self.regions_seg[i][k].at[index,'minor_axis_length'], 'coords' : xcoords, 'orientation' : xorientation, 'eccentricity' : self.regions_seg[i][k].at[index,'eccentricity'] }, ignore_index = True )
			
			t = trackpy.link( f, self.tolerance_tracking, memory = self.memory_tracking, adaptive_stop = self.adaptive_stop_tracking, adaptive_step = self.adaptive_step_tracking )
			self.cell_tracking[i] = t

	def cellsFinding ( self ):

		for i in range( len(self.stack_seg) ):
			for key in self.cell_tracking[i].groupby(by='particle').groups.keys():
				only_bacteria = {}
				only_bacteria['index'] = key
				only_bacteria['path'] = []
				for index in self.cell_tracking[i].groupby(by='particle').get_group( key ).index:
					only_bacteria['path'].append( { 
												'frame': self.cell_tracking[i].at[index,'frame'] , 
												'x' : self.cell_tracking[i].at[index,'x'], 
												'y': self.cell_tracking[i].at[index,'y'], 
												'index_data_frame' : int(self.cell_tracking[i].at[index,'index_frame']), 
												'index_tracking' : index, 
												'major_axis_length' : self.regions_seg[i][self.cell_tracking[i].at[index,'frame']].at[ int(self.cell_tracking[i].at[index,'index_frame']),'major_axis_length'] } )

				self.find_cells[i].append( only_bacteria )
		
		for i in range( len(self.stack_seg) ):
			self.cell_tracking[i].loc[:,'cell'] = -1

		for k in range( len(self.stack_seg) ):
			xdistance = 20
			xcell = 0
			for xrow in self.find_cells[k]:
				xdata = []
				if len(xrow['path']) >= (xdistance/2) : 
					for i in range( len(xrow['path']) ):
						xdata.append([ i, xrow['path'][i]['major_axis_length'] ])

					peaks, prop = scipy.signal.find_peaks( numpy.array(xdata)[:,1], height = (self.mean_length_cell), distance = xdistance )

					for i in range( len(peaks) ) :
						if i ==  0 :
							xindex = numpy.arange(0,peaks[i]+1)
						else:
							xindex = numpy.arange(peaks[i-1]+1,peaks[i]+1)

						for j in xindex:
							self.cell_tracking[k].loc[ xrow['path'][j]['index_tracking'] ,'cell'] = xcell

						if self.is_enabled_half_path_detection :
							if i == 0 and (peaks[i]+1) >= xdistance :
								self.only_bacteria_enabled[k].append( xcell )
							elif i > 0 and (peaks[i] - peaks[i-1]) >= xdistance :
								self.only_bacteria_enabled[k].append( xcell )

						xcell = xcell + 1
			
			for key in self.cell_tracking[k].query(' cell < 0  ').groupby(by='particle').groups.keys():
				for index in self.cell_tracking[k].query(' cell < 0  ').groupby(by='particle').get_group( key ).index:
					self.cell_tracking[k].loc[ index ,'cell'] = xcell
				xcell = xcell + 1

		for i in range( len(self.stack_seg) ):
		
			for key in self.cell_tracking[i].groupby(by='cell').groups.keys():
				only_bacteria = {}
				only_bacteria['index'] = key
				only_bacteria['path'] = []
				for index in self.cell_tracking[i].groupby(by='cell').get_group( key ).index:
					only_bacteria['path'].append( { 
												'frame': self.cell_tracking[i].at[index,'frame'] , 
												'x' : self.cell_tracking[i].at[index,'x'], 
												'y': self.cell_tracking[i].at[index,'y'], 
												'index_data_frame' : int(self.cell_tracking[i].at[index,'index_frame']), 
												'major_axis_length' : self.cell_tracking[i].at[index,'major_axis_length'],
												'minor_axis_length' : self.cell_tracking[i].at[index,'minor_axis_length'],
												'coords' : self.cell_tracking[i].at[index,'coords'],
												'orientation' : self.cell_tracking[i].at[index,'orientation'],
												'eccentricity' : self.cell_tracking[i].at[index,'eccentricity']
												} )

				self.find_bacterium[i].append( only_bacteria )

	def processExplore ( self ):

		for i in range( len(self.stack_seg) ):
			self.explore[i] = numpy.sum( numpy.sum( numpy.cumsum( self.stack_seg[i], axis = 0 ) > 0, axis = 1 ), axis = 1 )
			self.area[i] = numpy.sum( numpy.sum( self.stack_seg[i] > 0, axis = 1 ), axis = 1 )

	def processPath ( self ):

		micron_px = self.micron_px
		if type(self.micron_px) is int or type(self.micron_px) is float :
			micron_px = [micron_px]*len(self.files)

		for i in range( len(self.stack_seg) ):
			
			for row in self.find_bacterium[i]:
				if len(row['path']) > 1 :
					distance = 0
					path = 0

					distance = math.sqrt( math.pow( row['path'][len(row['path'])-1]['x'] - row['path'][0]['x'], 2 ) + math.pow( row['path'][len(row['path'])-1]['y'] - row['path'][0]['y'], 2 ) )*micron_px[i]
					deltaY = (row['path'][len(row['path'])-1]['y'] - row['path'][0]['y'])*micron_px[i]
					deltaX = (row['path'][len(row['path'])-1]['x'] - row['path'][0]['x'])*micron_px[i]
					points = []
					path_det = []
					points.append( [ row['path'][0]['x']*micron_px[i], row['path'][0]['y']*micron_px[i] ] )
					path_det.append([ row['path'][0]['frame'], row['path'][0]['major_axis_length']*micron_px[i], row['path'][0]['minor_axis_length']*micron_px[i], row['path'][0]['x']*micron_px[i], row['path'][0]['y']*micron_px[i], math.nan, math.nan, math.nan, row['path'][0]['orientation'] ])
					for index_path in range(1,len(row['path'])):
						dtheta = math.atan( (row['path'][index_path]['y']-row['path'][index_path-1]['y'])/(row['path'][index_path]['x']-row['path'][index_path-1]['x']) )*180/math.pi
						xdtheta = dtheta*math.pi/180
						if row['path'][index_path]['x'] > row['path'][index_path-1]['x'] and row['path'][index_path]['y'] > row['path'][index_path-1]['y']:
							dtheta = dtheta
						elif row['path'][index_path]['x'] > row['path'][index_path-1]['x'] and row['path'][index_path]['y'] < row['path'][index_path-1]['y']:
							dtheta = 360 + dtheta
						elif row['path'][index_path]['x'] < row['path'][index_path-1]['x'] and row['path'][index_path]['y'] > row['path'][index_path-1]['y']:
							dtheta = 180 + dtheta
						elif row['path'][index_path]['x'] < row['path'][index_path-1]['x'] and row['path'][index_path]['y'] < row['path'][index_path-1]['y']:
							dtheta = 180 + dtheta
						
						path = path + math.sqrt( math.pow( row['path'][index_path]['x'] - row['path'][index_path-1]['x'], 2 ) + math.pow( row['path'][index_path]['y'] - row['path'][index_path-1]['y'], 2 ) )*micron_px[i]
						points.append( [ row['path'][index_path]['x']*micron_px[i], row['path'][index_path]['y']*micron_px[i] ] )
						path_det.append([ row['path'][index_path]['frame'], row['path'][index_path]['major_axis_length']*micron_px[i], row['path'][index_path]['minor_axis_length']*micron_px[i], row['path'][index_path]['x']*micron_px[i], row['path'][index_path]['y']*micron_px[i], math.nan, math.nan, math.nan, row['path'][index_path]['orientation'] ])

						first_x_cell = 0
						first_y_cell = 0
						last_x_cell = 0
						last_y_cell = 0
						if row['path'][index_path-1]['orientation']*180/math.pi >= 0 :
							first_x_cell = numpy.max( row['path'][index_path-1]['coords'][:,1] )
							first_y_cell = numpy.max( row['path'][index_path-1]['coords'][:,0] )
							last_x_cell = numpy.min( row['path'][index_path-1]['coords'][:,1] )
							last_y_cell = numpy.min( row['path'][index_path-1]['coords'][:,0] )
						else:
							first_x_cell = numpy.min( row['path'][index_path-1]['coords'][:,1] )
							first_y_cell = numpy.max( row['path'][index_path-1]['coords'][:,0] )
							last_x_cell = numpy.max( row['path'][index_path-1]['coords'][:,1] )
							last_y_cell = numpy.min( row['path'][index_path-1]['coords'][:,0] )				

						maxdist = math.sqrt( math.pow( row['path'][index_path]['x'] - first_x_cell, 2 ) + math.pow( row['path'][index_path]['y'] - first_y_cell, 2 ) )*micron_px[i]
						mindist = math.sqrt( math.pow( row['path'][index_path]['x'] - last_x_cell, 2 ) + math.pow( row['path'][index_path]['y'] - last_y_cell, 2 ) )*micron_px[i]
						
						#dtheta_cellbody = math.nan
						dtheta_cellbody = row['path'][index_path-1]['orientation']*180/math.pi
						if maxdist > mindist:
							"""
							ratio_cellbody = (last_y_cell - first_y_cell)/(last_x_cell - first_x_cell)
							dtheta_cellbody = math.atan( ratio_cellbody )*180/math.pi
							if last_x_cell > first_x_cell and last_y_cell > first_y_cell :
								dtheta_cellbody = dtheta_cellbody
							elif last_x_cell > first_x_cell and last_y_cell < first_y_cell :
								dtheta_cellbody = 360 + dtheta_cellbody
							elif last_x_cell < first_x_cell and last_y_cell > first_y_cell :
								dtheta_cellbody = 180 + dtheta_cellbody
							elif last_x_cell < first_x_cell and last_y_cell < first_y_cell :
								dtheta_cellbody = 180 + dtheta_cellbody
							"""
							
							dtheta_cellbody = row['path'][index_path-1]['orientation']*180/math.pi
							dtheta_cellbody = (180 + dtheta_cellbody) if row['path'][index_path-1]['orientation']*180/math.pi >= 0 else (360 + dtheta_cellbody)
							
						elif mindist > maxdist:
							"""
							ratio_cellbody = (first_y_cell - last_y_cell)/(first_x_cell - last_x_cell)
							dtheta_cellbody = math.atan( ratio_cellbody )*180/math.pi
							if first_x_cell > last_x_cell and first_y_cell > last_y_cell :
								dtheta_cellbody = dtheta_cellbody
							elif first_x_cell > last_x_cell and first_y_cell < last_y_cell :
								dtheta_cellbody = 360 + dtheta_cellbody
							elif first_x_cell < last_x_cell and first_y_cell > last_y_cell :
								dtheta_cellbody = 180 + dtheta_cellbody
							elif first_x_cell < last_x_cell and first_y_cell < last_y_cell :
								dtheta_cellbody = 180 + dtheta_cellbody
							"""
							
							dtheta_cellbody = row['path'][index_path-1]['orientation']*180/math.pi
							dtheta_cellbody = dtheta_cellbody if row['path'][index_path-1]['orientation']*180/math.pi >= 0 else (180 + dtheta_cellbody)

						elif mindist == maxdist:
							dtheta_cellbody == row['path'][index_path-2]['orientation']

						path_det[index_path-1][5] = dtheta
						path_det[index_path-1][6] = dtheta_cellbody
						path_det[index_path-1][7] = xdtheta

					
					xcircle = smallestenclosingcircle.make_circle(points)
					
					dtheta = math.atan( deltaY/deltaX )*180/math.pi
					if row['path'][len(row['path'])-1]['x'] > row['path'][0]['x'] and row['path'][len(row['path'])-1]['y'] > row['path'][0]['y']:
						dtheta = dtheta
					elif row['path'][len(row['path'])-1]['x'] > row['path'][0]['x'] and row['path'][len(row['path'])-1]['y'] < row['path'][0]['y']:
						dtheta = 360 + dtheta
					elif row['path'][len(row['path'])-1]['x'] < row['path'][0]['x'] and row['path'][len(row['path'])-1]['y'] > row['path'][0]['y']:
						dtheta = 180 + dtheta
					elif row['path'][len(row['path'])-1]['x'] < row['path'][0]['x'] and row['path'][len(row['path'])-1]['y'] < row['path'][0]['y']:
						dtheta = 180 + dtheta
					
					if self.is_enabled_half_path_detection :
						if row['index'] in self.only_bacteria_enabled[i] :
							self.path_details[i].append( path_det )
							self.path_global[i].append([ row['index'], len(row['path']), path, distance, xcircle[2]*2, row['path'][ len(row['path']) - 1 ]['frame'] , deltaX, deltaY, dtheta ] )
					else:
						self.path_details[i].append( path_det )
						self.path_global[i].append([ row['index'], len(row['path']), path, distance, xcircle[2]*2, row['path'][ len(row['path']) - 1 ]['frame'] , deltaX, deltaY, dtheta ] )

										
	def setMaxImageExplorer ( self, max_images_explorer ):
		self.max_images_explorer = max_images_explorer

	def setIsKeepMultipleExplorer ( self, is_keep_multiple_explorer = False ):
		self._is_keep_multiple_explorer = is_keep_multiple_explorer

	def getExplore ( self ):
		xarea = []
		xexplore = []
		micron_px = self.micron_px
		if type(self.micron_px) is int or type(self.micron_px) is float:
			micron_px = [micron_px]*len(self.files)
		
		for i in range( len(self.stack_seg) ):
			ximages = self.stack_seg[i].shape[0]
			xa = self.area[i][1:self.max_images_explorer]*(micron_px[i]**2)
			xe = (self.explore[i][1:self.max_images_explorer]*(micron_px[i]**2) - self.explore[i][0:self.max_images_explorer-1]*(micron_px[i]**2))/self.time_seq

			xarea.append( xa )
			xexplore.append( xe )

		if self._is_keep_multiple_explorer :
			return xarea, xexplore
		else:
			return numpy.hstack( (xarea) ), numpy.hstack( xexplore )

	def getAllPathDetails ( self ):
		xdetails = []
		for i in range( len(self.stack_seg) ):
			for path in self.path_details[i]:
				xdetails.append( path )

		return xdetails

	def getAllPathGlobal ( self ):
		xglobal = []
		for i in range( len(self.stack_seg) ):
			for path in self.path_global[i]:
				xglobal.append( path )

		return xglobal

	def showCumStack ( self ):
		plt.figure( figsize = (40, 8) )
		for i in range( len(self.stack_seg) ):
			plt.subplot(1, len(self.stack_seg), i+1)
			plt.imshow( ( numpy.sum( self.stack_seg[i], axis = 0 ) > 0 ), cmap = 'gray' )

	def showDetectedCells ( self ):
		plt.figure( figsize = (40, 8) )
		for i in range( len(self.stack_seg) ):
			plt.subplot(1, len(self.stack_seg), i+1)
			plt.scatter( self.cell_tracking[i]['x'], self.cell_tracking[i]['y'] )
			plt.ylim( [0, self.stack_seg[i].shape[1] ] )
			plt.xlim( [0, self.stack_seg[i].shape[2] ] )
			plt.axis( 'scaled' )

	def showCellsTracks ( self ):
		plt.figure( figsize = (40, 8) )
		for i in range( len(self.stack_seg) ):
			plt.subplot(1, len(self.stack_seg), i+1)
			for key in self.cell_tracking[i].groupby(by='particle').groups.keys():
				plt.plot( self.cell_tracking[i].groupby(by='particle').get_group( key )['x'].tolist(), self.cell_tracking[i].groupby(by='particle').get_group( key )['y'].tolist() )
				plt.ylim( [0, self.stack_seg[i].shape[1] ] )
				plt.xlim( [0, self.stack_seg[i].shape[2] ] )
				plt.axis( 'scaled' )

	class resource :

		def __init__ ( self ):
			self.time_area = []
			self.time_explorer = []
			self.is_multiple = False

		def setTimeArea ( self, time_area ):
			self.time_area = time_area

		def setTimeExplorer ( self, time_explorer ):
			self.time_explorer = time_explorer

		def __fit_global_motility ( self ):

			def funct_fit(A, m):
				return m*A

			if self.is_multiple :
				xfit = []
				for i in range( len(self.time_area) ):
					res_fit = curve_fit( funct_fit, self.time_area[i], self.time_explorer[i] )
					xfit.append( res_fit[0][0] )
				return xfit
			else:
				res_fit = curve_fit( funct_fit, self.time_area, self.time_explorer )
				return res_fit[0][0]

		def getFitExplorer ( self ):
			return self.__fit_global_motility()

		def getGlobalSpeed ( self, division_time, cell_size ):
			result = self.__fit_global_motility()
			if self.is_multiple:
				xgs = []
				for fit in result:
					xgs.append( (fit-1/division_time)*cell_size )
				return xgs
			else:
				return (result-1/division_time)*cell_size

		def showScatterGlobalSpeed ( self, is_sub = False, color_name = 'black', marker = 'o', ls = '-' ):

			result = self.__fit_global_motility()

			if not is_sub :
				plt.figure( figsize = (7,5) )

			if self.is_multiple:
				xmarker = marker
				xcolor = color_name
				xls = ls
				if type(xmarker) is not list:
					xmarker = [xmarker]*len(result)
				if type(xcolor) is not list:
					xcolor = [xcolor]*len(result)
				if type(xls) is not list:
					xls = [xls]*len(result)
					
				for i in range( len(result) ):
					plt.plot( self.time_area[i], self.time_explorer[i], linestyle = '', marker = xmarker[i], color = xcolor[i], alpha = 0.2  )
					plt.plot( self.time_area[i], self.time_area[i]*result[i], color = xcolor[i], linestyle = xls[i] )
			else:
				plt.plot( self.time_area, self.time_explorer, linestyle = '', marker = marker, color = color_name, alpha = 0.2  )
				plt.plot( self.time_area, self.time_area*result, color = color_name, linestyle = ls )
			
			plt.xlabel(r'$A \; (\mu m^2)$',fontdict = {'size' : 16})
			plt.ylabel(r'$\Delta S / \Delta t \; (\mu m^2.min^{-1})$', fontdict = {'size' : 16})
			plt.xticks( fontsize = 16 )
			plt.yticks( fontsize = 16 )
			plt.grid( linestyle = ':' )

			return plt




