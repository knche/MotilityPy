from motility import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
import math


class multi_graph :

	def __init__ ( self, data = [], names = [], colors = [], micron_px = 1, time_seq = 1, time_acquisition = 0 ):

		self.motilitys = []
		self._font_size = 16
		self._names = names
		self._colors= colors
		self._interval_axis_plot = 6
		self._test_threshold_value = 0.01

		self._micron_px = micron_px
		if type(self._micron_px) is int or type(self._micron_px) is float :
			self._micron_px = [micron_px]*len(data)

		self._time_seq = time_seq
		if type(self._time_seq) is int or type(self._time_seq) is float :
			self._time_seq = [time_seq]*len(data)

		self._time_acquisition = time_acquisition
		if type(self._time_acquisition) is int or type(self._time_acquisition) is float :
			self._time_acquisition = [time_acquisition]*len(data)

		for i in range( len(data) ):
			self.motilitys.append( motility( data = data[i], micron_px = self._micron_px[i], time_seq = self._time_seq[i], time_acquisition = self._time_acquisition[i] ) )


	def run ( self ):

		for i in range( len(self.motilitys) ):
			self.motilitys[i].run()

	def setFontSize ( self, font_size = 16 ):
		self._font_size= font_size
	def setNames ( self, names  = [] ):
		self._names = names
	def setColors ( self, colors = [] ):
		self._colors = colors

	# tracking
	
	def showCellXTimeTracking ( self ):

		xtime = []
		xpoint = []
		for i in range( len(self.motilitys) ):
			time, point = self.motilitys[i].getAllXPath()
			xtime.append( time )
			xpoint.append( point )

		plt.figure( figsize = (6*len(xtime),4) )

		for i in range( len(xtime) ):
			plt.subplot(1, len(xtime), i+1)
			plt.plot( xtime[i], xpoint[i], color = self._colors[i] )
			plt.xlabel('t (min)',fontdict = {'size' : self._font_size})
			plt.ylabel(r'$x \; (\mu m)$', fontdict = {'size' : self._font_size})
			plt.xticks( fontsize = self._font_size )
			plt.yticks( fontsize = self._font_size )
			plt.grid( linestyle = ':' )

		plt.tight_layout()

		return plt

	def showCellYTimeTracking ( self ):

		ytime = []
		ypoint = []
		for i in range( len(self.motilitys) ):
			time, point = self.motilitys[i].getAllYPath()
			ytime.append( time )
			ypoint.append( point )

		plt.figure( figsize = (6*len(ytime),4) )

		for i in range( len(ytime) ):
			plt.subplot(1, len(ytime), i+1)
			plt.plot( ytime[i], ypoint[i], color = self._colors[i] )
			plt.xlabel('t (min)',fontdict = {'size' : self._font_size})
			plt.ylabel(r'$y \; (\mu m)$', fontdict = {'size' : self._font_size})
			plt.xticks( fontsize = self._font_size )
			plt.yticks( fontsize = self._font_size )
			plt.grid( linestyle = ':' )

		plt.tight_layout()

		return plt

	def showCellXYTracking ( self ):

		xtime = []
		xpoint = []
		ytime = []
		ypoint = []
		for i in range( len(self.motilitys) ):
			tx, ptx = self.motilitys[i].getAllXPath()
			ty, pty = self.motilitys[i].getAllYPath()
			xtime.append( tx )
			xpoint.append( ptx )
			ytime.append( ty )
			ypoint.append( pty )


		plt.figure( figsize = (6*len(xpoint),4) )

		for i in range( len(xpoint) ):
			plt.subplot(1, len(xpoint), i+1)
			plt.plot( xpoint[i] , ypoint[i] , color = self._colors[i] )
			plt.xlabel(r'$x \; (\mu m)$',fontdict = {'size' : self._font_size})
			plt.ylabel(r'$y \; (\mu m)$', fontdict = {'size' : self._font_size})
			plt.xticks( fontsize = self._font_size )
			plt.yticks( fontsize = self._font_size )
			plt.axis( 'equal' )
			plt.grid( linestyle = ':' )

		plt.tight_layout()

		return plt

	# characteristics

	def showGlobalDivisionTime ( self, is_sub = False ):

		divtime = []
		for i in range( len(self.motilitys) ):
			divtime.append( self.motilitys[i].getDivisionTime() )

		if not is_sub :
			plt.figure( figsize = (2*len(divtime), 5) )
		
		span = 0.2
		index_position = []
		for i in range( len(divtime) ):
			plt.boxplot( x = [ divtime[i] ], positions = [1 + span*i], widths=0.1, notch = True, patch_artist = True, boxprops = dict( facecolor = self._colors[i] ), medianprops = dict( linewidth = 1, color = 'black' ) )			
			index_position.append( 1 + span*i )

		plt.xticks( index_position, self._names, fontsize = self._font_size, rotation = 45 )
		plt.yticks( fontsize = self._font_size )
		plt.ylabel('Division time (min)', fontdict = { 'size' : self._font_size })
		plt.grid( linestyle = ':' )

		return plt

	def showGlobalCellSize ( self, is_sub = False ):

		csize = []
		for i in range( len(self.motilitys) ):
			csize.append( self.motilitys[i].getCellSize() )

		if not is_sub :
			plt.figure( figsize = (2*len(csize), 5) )

		span = 0.2
		index_position = []
		for i in range( len(csize) ):
			plt.boxplot( x = [ csize[i] ], positions = [1 + span*i], widths=0.1, notch = True, patch_artist = True, boxprops = dict( facecolor = self._colors[i] ), medianprops = dict( linewidth = 1, color = 'black' ) )
			index_position.append( 1 + span*i )

		plt.xticks( index_position, self._names, fontsize = self._font_size, rotation = 45 )
		plt.yticks( fontsize = self._font_size )
		plt.ylabel('Cell size ($ \mu m $)', fontdict = { 'size' : self._font_size })
		plt.grid( linestyle = ':' )

		return plt

	def showGlobalAspectRatio ( self, is_sub = False ):

		aratio = []
		for i in range( len(self.motilitys) ):
			aratio.append( self.motilitys[i].getAspectRatio() )

		if not is_sub :
			plt.figure( figsize = (2*len(aratio), 5) )

		span = 0.2
		index_position = []
		for i in range( len(aratio) ):
			plt.boxplot( x = [ aratio[i] ], positions = [1 + span*i], widths=0.1, notch = True, patch_artist = True, boxprops = dict( facecolor = self._colors[i] ), medianprops = dict( linewidth = 1, color = 'black' ) )
			index_position.append( 1 + span*i )

		plt.xticks( index_position, self._names, fontsize = self._font_size, rotation = 45 )
		plt.yticks( fontsize = self._font_size )
		plt.ylabel('Aspect ratio', fontdict = { 'size' : self._font_size })
		plt.grid( linestyle = ':' )

		return plt

	def showGlobalGrowthRate ( self, is_sub = False ):

		grate = []
		for i in range( len(self.motilitys) ):
			grate.append( self.motilitys[i].getGrowthRate() )

		if not is_sub :
			plt.figure( figsize = (2*len(grate), 6) )

		span = 0.2
		index_position = []
		for i in range( len(grate) ):
			plt.boxplot( x = [ grate[i] ], showfliers = False, positions = [1 + span*i], widths=0.1, notch = True, patch_artist = True, boxprops = dict( facecolor = self._colors[i] ), medianprops = dict( linewidth = 1, color = 'black' ) )
			index_position.append( 1 + span*i )

		plt.xticks( index_position, self._names, fontsize = self._font_size, rotation = 45 )
		plt.yticks( fontsize = self._font_size )
		plt.ylabel('Growth rate ($\mu m.min^{-1}$)', fontdict = { 'size' : self._font_size })
		plt.grid( linestyle = ':' )

		return plt

	def showGroupDivTimCSizeARGroRate ( self ):

		plt.figure( figsize = (1.4*len(self.motilitys)*4,5) )

		plt.subplot(141)
		self.showGlobalDivisionTime( is_sub = True )

		plt.subplot(142)
		self.showGlobalCellSize( is_sub = True )

		plt.subplot(143)
		self.showGlobalAspectRatio( is_sub = True )

		plt.subplot(144)
		self.showGlobalGrowthRate( is_sub = True )

		plt.tight_layout()

		return plt

	# details

	def showAverageDistancePersistence ( self, show_std = True ):
		timepersistence = []
		avgpersistence = []
		stdpersistence = []
		for i in range( len(self.motilitys) ):
			time, average, std = self.motilitys[i].getAveragePersistenceDistance()		
			timepersistence.append( time )
			avgpersistence.append( average )
			stdpersistence.append( std )

		plt.figure( figsize = (12,5) )

		for i in range( len(timepersistence) ):
			plt.plot( timepersistence[i], avgpersistence[i], color = self._colors[i], label = self._names[i] )
			if show_std :
				plt.fill_between( timepersistence[i], avgpersistence[i] - stdpersistence[i], avgpersistence[i] + stdpersistence[i], color = self._colors[i], alpha = 0.2 )

		plt.xticks( fontsize = self._font_size )
		plt.yticks( fontsize = self._font_size )
		plt.xlabel('Time (min)', fontdict = { 'size' : self._font_size })
		plt.ylabel('Persistence', fontdict = { 'size' : self._font_size })
		plt.grid( linestyle = ':' )
		plt.legend( frameon = False )		

		return plt

	# global lengths

	def showGlobalEndtoEndDistance ( self, is_test = False ):

		endtoenddistance = []
		for i in range( len(self.motilitys) ):
			endtoenddistance.append( self.motilitys[i].getGlobalEndtoEndDistance() )

		return self.__scheme_single_boxplot( data = endtoenddistance, ylabel = 'Net distance traveled ($\mu m$)', is_test = is_test )

	def showGlobalPath ( self, is_test = False ):

		pathtotal = []
		for i in range( len(self.motilitys) ):
			pathtotal.append( self.motilitys[i].getGlobalPath() )

		return self.__scheme_single_boxplot( data = pathtotal, ylabel = 'Total distance traveled ($\mu m$)', is_test = is_test )

	def showGlobalBlobDiameter ( self, is_test = False ):

		blobdiameter = []
		for i in range( len(self.motilitys) ):
			blobdiameter.append( self.motilitys[i].getGlobalDiameterBlob() )

		return self.__scheme_single_boxplot( data = blobdiameter, ylabel = 'Blob diameter ($\mu m$)', is_test = is_test )

	def showGlobalMaxDistance ( self, is_test = False ):

		maxdistance = []
		for i in range( len(self.motilitys) ):
			maxdistance.append( self.motilitys[i].getGlobalMaxDistance() )

		return self.__scheme_single_boxplot( data = maxdistance, ylabel = 'Max distance traveled ($\mu m$)', is_test = is_test )		

	def showGlobalAllLengths ( self, is_test = False ):

		lengths = []
		for i in range( len(self.motilitys) ):
			path = self.motilitys[i].getGlobalPath()		
			endtoend = self.motilitys[i].getGlobalEndtoEndDistance()
			blobd = self.motilitys[i].getGlobalDiameterBlob()
			maxd = self.motilitys[i].getGlobalMaxDistance()

			lengths.append([ path, endtoend, blobd, maxd ])

		return self.__scheme_multiple_boxplot( data = lengths, xlabels = ['Total distance traveled','Net distance traveled','Blob diameter','Max distance traveled'], ylabel = 'Length ($\mu m$)', is_legend = True )


	# global ratios

	def showGlobalPersistenceRatio ( self, is_test = False ):

		persistenceratio = []
		for i in range( len(self.motilitys) ):
			persistenceratio.append( self.motilitys[i].getGlobalPersistenceRatio() )

		return self.__scheme_single_boxplot( data = persistenceratio, ylabel = 'Persistence ratio', is_test = is_test )

	def showGlobalDirectionalityRatio ( self, is_test = False ):

		directratio = []
		for i in range( len(self.motilitys) ):
			directratio.append( self.motilitys[i].getGlobalDirectionalityRatio() )

		return self.__scheme_single_boxplot( data = directratio, ylabel = 'Directionality ratio', is_test = is_test )

	def showGlobalDisplacementRatio ( self, is_test = False ):

		displacratio = []
		for i in range( len(self.motilitys) ):
			displacratio.append( self.motilitys[i].getGlobalDisplacementRatio() )

		return self.__scheme_single_boxplot( data = directratio, ylabel = 'Displacement ratio', is_test = is_test )

	def showGlobalOutreachRatio ( self, is_test = False ):

		outreachratio = []
		for i in range( len(self.motilitys) ):
			outreachratio.append( self.motilitys[i].getGlobalOutreachRatio() )

		return self.__scheme_single_boxplot( data = outreachratio, ylabel = 'Outreach ratio', is_test = is_test )		

	def showGlobalExplorerRatio ( self, is_test = False ):

		explorerratio = []
		for i in range( len(self.motilitys) ):
			explorerratio.append( self.motilitys[i].getGlobalExplorerRatio() )

		return self.__scheme_single_boxplot( data = explorerratio, ylabel = 'Explorer ratio', is_test = is_test )

	# global speeds

	def showGlobalMeanStraightLineSpeed ( self, is_test = False ):

		avgspeed = []
		for i in range( len(self.motilitys) ):
			avgspeed.append( self.motilitys[i].getGlobalMeanStraightLineSpeed() )

		return self.__scheme_single_boxplot( data = avgspeed, ylabel = 'Mean Straight Line Speed ($\mu m / min$)', is_test = is_test )

	def showGlobalTotalSpeed ( self, is_test = False ) :

		totalspeed = []
		for i in range( len(self.motilitys) ):
			totalspeed.append( self.motilitys[i].getGlobalTotalSpeed() )
		
		return self.__scheme_single_boxplot( data = totalspeed, ylabel = 'Total Speed ($\mu m / min$)', is_test = is_test )

	def showGlobalMeanCurvilinearSpeed ( self, is_test = False ):

		avgcurvspeed = []
		for i in range( len(self.motilitys) ):
			avgcurvspeed.append( self.motilitys[i].getGlobalMeanCurvilinearSpeed() )

		return self.__scheme_single_boxplot( data = avgcurvspeed, ylabel = 'Mean Curvilinear Speed ($\mu m / min$)', is_test = is_test )

	def showGlobalGSpeed ( self ):

		gspeed = []
		stdgspeed = []
		index_position = []
		for i in range( len(self.motilitys) ):
			gspeed.append( self.motilitys[i].getGSpeed() )
			stdgspeed.append( self.motilitys[i].getStdGSpeed() )
			index_position.append( i + 1 )

		plt.bar( x = index_position, height = gspeed, yerr = stdgspeed, color = self._colors[i] )
		plt.ylabel(r'$V_{g} \; (\mu m.min^{-1})$',fontdict = {'size' : self._font_size})
		plt.xticks( index_position, self._names, fontsize = self._font_size, rotation = 45 )
		plt.yticks( fontsize = self._font_size )
		plt.grid( linestyle = ':' )

		return plt

	# util

	def __scheme_single_boxplot ( self, data = [], ylabel = '', span = 0.2, is_test = False, spancap = 2.1 ):

		index_position = []
		plt.figure( figsize = ( 1.5*len(data) , 5) )

		for i in range( len(data) ):
			plt.boxplot( x = [ data[i] ], positions = [ 1 + span*i ], widths=0.1, notch = True, patch_artist = True, boxprops = dict( facecolor = self._colors[i] ), medianprops = dict( linewidth = 1, color = 'black' ) )
			index_position.append( 1 + span*i )
		
		plt.xticks( index_position, self._names, fontsize = self._font_size, rotation = 45 )
		plt.yticks( fontsize = self._font_size )
		plt.ylabel( ylabel, fontdict = { 'size' : self._font_size })
		plt.grid( linestyle = ':' )

		if is_test :

			mcomb = numpy.nonzero( numpy.tri( len(data) , len(data)  , -1) )
			mpair = numpy.dstack( (mcomb[1],mcomb[0]) )[0]
			pvalues = []
			count_true_test = 0
			for row in mpair:
				test_pvalue = stats.ttest_ind( data[ row[0] ], data[ row[1] ] ).pvalue
				pvalues.append( test_pvalue )
				if test_pvalue < self._test_threshold_value:
					count_true_test = count_true_test + 1

			if count_true_test > 0:

				lims = plt.axis()
				y_inter = ( lims[3] - lims[2] )/self._interval_axis_plot
				plt.ylim([ lims[2], lims[3] + y_inter ])

				spany = y_inter/count_true_test
				spanx = span/2
				for i in range( len( pvalues ) ):
					if pvalues[i] < self._test_threshold_value:
						difspancap = abs( mpair[i][0] - mpair[i][1] )

						plt.annotate('**', xy=(1 + span*mpair[i][0] + spanx*difspancap , lims[3] + y_inter - spany*i), ha='center', va='bottom', xycoords = 'data', arrowprops=dict(arrowstyle='-[, widthB='+str( spancap*difspancap )+', lengthB=0.5') )

		return plt

	def __scheme_multiple_boxplot ( self, data = [], xlabels = [], span = 0.2, ylabel = '', is_test = False, is_legend = False ):

		plt.figure( figsize = (1.5*len(data[0])*len(data), 5) )

		index_position  = [ ]
		xticks_position = []
		for i in range( len(data) ):
			positions = []
			for j in range( len(data[i]) ):
				positions.append( j + 1 + span*i )
			index_position.append( positions )
			plt.boxplot( x = data[i], positions = positions, widths=0.1, notch = True, patch_artist = True, boxprops = dict( facecolor = self._colors[i] ), medianprops = dict( linewidth = 1, color = 'black' ) )

		for i in range( len(index_position[0]) ):
			tickinter = abs(index_position[ len(index_position) - 1 ][0] - index_position[0][0])
			xticks_position.append( index_position[0][i] + tickinter/( len(index_position) - 1 ) )

		plt.xticks( xticks_position, xlabels, fontsize = self._font_size )
		plt.yticks( fontsize = self._font_size )
		plt.ylabel( ylabel , fontdict = { 'size' : self._font_size })
		plt.grid( linestyle = ':' )
		
		if is_legend :

			legend = []
			for i in range( len(data) ):
				legend.append( Line2D([0], [0], color = self._colors[i] ) )
			
			plt.legend( legend, self._names, loc = 'upper right', frameon = False )

		return plt

