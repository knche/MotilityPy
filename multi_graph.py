from motility import *
from util import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math
import numpy
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic, linregress
from colour import Color

class multi_graph :

	def __init__ ( self, data = [], names = [], colors = [], micron_px = 1, time_seq = 1, time_acquisition = 0 ):

		self.motilitys = []
		self._font_size = 16
		self._names = names
		self._colors = colors
		self._interval_axis_plot = 6
		self._test_threshold_value = 0.01

		self._util = util ( font_size = 16, colors = colors, names = names, interval_axis_plot = 6, test_threshold_value = 0.01 )

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
		self._util.setFontSize( font_size = font_size )
	def setNames ( self, names  = [] ):
		self._names = names
		self._util.setNames( names = names )
	def setColors ( self, colors = [] ):
		self._colors = colors
		self._util.setColors( colors = colors )
	def setIntervalAxisPlot ( self, interval_axis_plot = 6 ):
		self._interval_axis_plot = interval_axis_plot
		self._util.setIntervalAxisPlot( interval_axis_plot )

	# utils

	def __UtilClearXY ( self, xdata, ydata ) :

		xtmp_data = xdata
		ytmp_data = ydata

		i_nonnan = numpy.where( ~numpy.isnan(xtmp_data) )[0]
		xtmp_data = xtmp_data[i_nonnan]
		ytmp_data = ytmp_data[i_nonnan]

		i_nonnan = numpy.where( ~numpy.isnan(ytmp_data) )[0]
		xtmp_data = xtmp_data[i_nonnan]
		ytmp_data = ytmp_data[i_nonnan]

		i_noninf = numpy.where( ~numpy.isinf(xtmp_data) )[0]
		xtmp_data = xtmp_data[i_noninf]
		ytmp_data = ytmp_data[i_noninf]

		i_noninf = numpy.where( ~numpy.isinf(ytmp_data) )[0]
		xtmp_data = xtmp_data[i_noninf]
		ytmp_data = ytmp_data[i_noninf]

		return xtmp_data, ytmp_data

	# tracking
	
	def showCellXTimeTracking ( self, is_color = False, is_legend = False ):

		xtime = []
		xpoint = []
		for i in range( len(self.motilitys) ):
			time, point = self.motilitys[i].getAllXPath()
			xtime.append( time )
			xpoint.append( point )

		return self._util.scheme_plot_fill( xdata = xtime, ydata = xpoint, ystd = [], xlabel = 't (min)', ylabel = '$x \; (\mu m)$', size = (6*len(xpoint),5), is_multi = True, is_color = is_color, is_legend = is_legend )

	def showCellYTimeTracking ( self, is_color = False, is_legend = False ):

		ytime = []
		ypoint = []
		for i in range( len(self.motilitys) ):
			time, point = self.motilitys[i].getAllYPath()
			ytime.append( time )
			ypoint.append( point )

		return self._util.scheme_plot_fill( xdata = ytime, ydata = ypoint, ystd = [], xlabel = 't (min)', ylabel = '$y \; (\mu m)$', size = (6*len(ypoint),5), is_multi = True, is_color = is_color, is_legend = is_legend )

	def showCellXYTracking ( self, is_color = False, is_legend = False ):

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

		return self._util.scheme_plot_fill( xdata = xpoint, ydata = ypoint, ystd = [], xlabel = r'$x \; (\mu m)$', ylabel = '$y \; (\mu m)$', size = (6*len(xpoint),5), is_multi = True, is_axis_equal = True, is_color = is_color, is_legend = is_legend )
	
	def showAverageXTime ( self, is_multi = False, is_legend = False ):
		
		time = []
		average = []
		std = []
		for i in range( len(self.motilitys) ):
			xtime, xavg, xstd = self.motilitys[i].getAverageX()
			time.append( xtime )
			average.append( xavg )
			std.append( xstd )

		xsize = (6*len(self.motilitys),5) if is_multi else (6,5)
		return self._util.scheme_plot_fill( xdata = time, ydata = average, ystd = std, xlabel = 't (min)', ylabel = '$x \; (\mu m)$', size = xsize, is_multi = is_multi, is_legend = is_legend )

	def showAverageYTime ( self, is_multi = False, is_legend = False ):
		
		time = []
		average = []
		std = []
		for i in range( len(self.motilitys) ):
			xtime, xavg, xstd = self.motilitys[i].getAverageY()
			time.append( xtime )
			average.append( xavg )
			std.append( xstd )

		xsize = (6*len(self.motilitys),5) if is_multi else (6,5)
		return self._util.scheme_plot_fill( xdata = time, ydata = average, ystd = std, xlabel = 't (min)', ylabel = '$y \; (\mu m)$', size = xsize, is_multi = is_multi, is_legend = is_legend )

	# difference tracking

	def showCellDifferenceXTimeTracking ( self, is_color = False, is_legend = False ):

		xtime = []
		xpoint = []
		for i in range( len(self.motilitys) ):
			time, point = self.motilitys[i].getAllDeltaX()
			xtime.append( time )
			xpoint.append( point )

		return self._util.scheme_plot_fill( xdata = xtime, ydata = xpoint, ystd = [], xlabel = 't (min)', ylabel = '$\Delta x \; (\mu m)$', size = (6*len(xpoint),5), is_multi = True, is_color = is_color, is_legend = is_legend )

	def showCellDifferenceYTimeTracking (self, is_color = False, is_legend = False ):

		ytime = []
		ypoint = []
		for i in range( len(self.motilitys) ):
			time, point = self.motilitys[i].getAllDeltaY()
			ytime.append( time )
			ypoint.append( point )

		return self._util.scheme_plot_fill( xdata = ytime, ydata = ypoint, ystd = [], xlabel = 't (min)', ylabel = '$\Delta y \; (\mu m)$', size = (6*len(ypoint),5), is_multi = True, is_color = is_color, is_legend = is_legend )

	# statistical significance
	def showStatisticalSignificanceBySteps ( self, is_norm = False, is_multi = False, is_legend = False ):

		xsteps = []
		xpoint = []
		for i in range( len(self.motilitys) ):
			time, point = self.motilitys[i].getAllXPath()
			norm = numpy.sum( ~numpy.isnan(point), axis = 1 )
			if is_norm :
				norm = norm/numpy.sum(~numpy.isnan(point))

			xsteps.append( numpy.arange( norm[ norm > 0 ].shape[0] ) )
			xpoint.append( norm[ norm > 0 ] )


		xsize = (6*len(self.motilitys), 5) if is_multi else (6, 5)
		return self._util.scheme_plot_fill( xdata = xsteps, ydata = xpoint, xlabel = 'Steps', ylabel = 'Number of Tracks', size = xsize, is_multi = is_multi, is_legend = is_legend )

	# quartile X, Y displacement

	def showQuantileLinesDifferenceXTimeTracking ( self, quantile = [0.2,0.3,0.4,0.5,0.6,0.7,0.8], is_legend = False ):
		
		xtime = []
		xpoint = []
		for i in range( len(self.motilitys) ):
			time, point = self.motilitys[i].getAllDeltaX()
			xtime.append( time )
			xpoint.append( numpy.nanquantile(point, quantile, axis = 1).T )

		return self._util.scheme_plot_fill( xdata = xtime, ydata = xpoint, ystd = [], xlabel = 't (min)', ylabel = 'Quantile', size = (6*len(xpoint),5), is_multi = True, is_color = True, is_legend = False )

	def showQuantileLinesDifferenceYTimeTracking ( self, quantile = [0.2,0.3,0.4,0.5,0.6,0.7,0.8], is_legend = False ):

		ytime = []
		ypoint = []
		for i in range( len(self.motilitys) ):
			time, point = self.motilitys[i].getAllDeltaY()
			ytime.append( time )
			ypoint.append( numpy.nanquantile(point, quantile, axis = 1).T )

		return self._util.scheme_plot_fill( xdata = ytime, ydata = ypoint, ystd = [], xlabel = 't (min)', ylabel = 'Quantile', size = (6*len(ypoint),5), is_multi = True, is_color = True, is_legend = False )

	def showHistDifferenceX ( self, dtype = 'hist', ylim = None, is_multi = False, marker = 'o', markersize = 10, is_fit = False, type_fit = 'linear', htype = 'bar', xscale = 'linear', yscale = 'linear', ylabel = 'Count', bins = 10, is_density = False, is_legend = False ):

		vanHoveX = []
		for i in range( len(self.motilitys) ):
			vanHoveX.append( self.motilitys[i].getAllDeltaX( is_flat= True ) )
		
		if dtype == 'hist' :
			return self._util.scheme_hist( data = vanHoveX, ylim = ylim, htype = htype, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$\Delta x \; (\mu m)$', density = is_density, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = vanHoveX, ylim = ylim, marker = marker, markersize = markersize, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$\Delta x \; (\mu m)$', density = is_density, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )

	def showHistDifferenceY ( self, dtype = 'hist', ylim = None, is_multi = False, marker = 'o', markersize = 10, is_fit = False, type_fit = 'linear', htype = 'bar', xscale = 'linear', yscale = 'linear', ylabel = 'Count', bins = 10, is_density = False, is_legend = False ):

		vanHoveY = []
		for i in range( len(self.motilitys) ):
			vanHoveY.append( self.motilitys[i].getAllDeltaY( is_flat= True ) )
		
		if dtype == 'hist':
			return self._util.scheme_hist( data = vanHoveY, ylim = None, htype = htype, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$\Delta y \; (\mu m)$', density = is_density, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = vanHoveY, ylim = None, marker = marker, markersize = markersize, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$\Delta y \; (\mu m)$', density = is_density, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )

	def showInterQuartileDifferenceX ( self, is_test = False, showfliers = True ):

		difference_x = []
		for i in range( len(self.motilitys) ):
			difference_x.append( self.motilitys[i].getAllDeltaX( is_flat= True ) )

		return self._util.scheme_single_boxplot( data = difference_x, ylabel = r'$\Delta x \; (\mu m)$', is_test = is_test, showfliers = showfliers )

	def showInterQuartileDifferenceY ( self, is_test = False, showfliers = True ):

		difference_y = []
		for i in range( len(self.motilitys) ):
			difference_y.append( self.motilitys[i].getAllDeltaY( is_flat= True ) )

		return self._util.scheme_single_boxplot( data = difference_y, ylabel = r'$\Delta y \; (\mu m)$', is_test = is_test, showfliers = showfliers )	

	def showHistAbsoluteDifferenceX ( self, dtype = 'hist', ylim = None, is_multi = False, marker = 'o', markersize = 10, is_fit = False, type_fit = 'linear', htype = 'bar', xscale = 'linear', yscale = 'linear', ylabel = 'Count', bins = 10, is_density = False, is_legend = False ):
		
		AbsX = []
		for i in range( len(self.motilitys) ):
			AbsX.append( numpy.absolute( self.motilitys[i].getAllDeltaX( is_flat= True ) ) )
		
		if dtype == 'hist' :
			return self._util.scheme_hist( data = AbsX, ylim = ylim, htype = htype, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$|\Delta x| \; (\mu m)$', density = is_density, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = AbsX, ylim = ylim, marker = marker, markersize = markersize, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$|\Delta x| \; (\mu m)$', density = is_density, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )

	def showHistAbsoluteDifferenceY ( self, dtype = 'hist', ylim = None, is_multi = False, marker = 'o', markersize = 10, is_fit = False, type_fit = 'linear', htype = 'bar', xscale = 'linear', yscale = 'linear', ylabel = 'Count', bins = 10, is_density = False, is_legend = False ):
		
		AbsY = []
		for i in range( len(self.motilitys) ):
			AbsY.append( numpy.absolute( self.motilitys[i].getAllDeltaY( is_flat= True ) ) )
		
		if dtype == 'hist' :
			return self._util.scheme_hist( data = AbsY, ylim = ylim, htype = htype, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$|\Delta y| \; (\mu m)$', density = is_density, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = AbsY, ylim = ylim, marker = marker, markersize = markersize, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$|\Delta y| \; (\mu m)$', density = is_density, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
	
	def showInterQuartileAbsoluteDifferenceX ( self, is_test = False, showfliers = True ):

		abs_difference_x = []
		for i in range( len(self.motilitys) ):
			abs_difference_x.append( numpy.absolute( self.motilitys[i].getAllDeltaX( is_flat= True ) ) )

		return self._util.scheme_single_boxplot( data = abs_difference_x, ylabel = r'$|\Delta x| \; (\mu m)$', is_test = is_test, showfliers = showfliers )

	def showInterQuartileAbsoluteDifferenceY ( self, is_test = False, showfliers = True ):

		abs_difference_y = []
		for i in range( len(self.motilitys) ):
			abs_difference_y.append( numpy.absolute( self.motilitys[i].getAllDeltaY( is_flat= True ) ) )

		return self._util.scheme_single_boxplot( data = abs_difference_y, ylabel = r'$|\Delta y| \; (\mu m)$', is_test = is_test, showfliers = showfliers )

	def showHistSquaredDifferenceX ( self, dtype = 'hist', ylim = None, is_multi = False, marker = 'o', markersize = 10, is_fit = False, type_fit = 'linear', htype = 'bar', xscale = 'linear', yscale = 'linear', ylabel = 'Count', bins = 10, is_density = False, is_legend = False ):

		SquaredX = []
		for i in range( len(self.motilitys) ):
			SquaredX.append( numpy.power( self.motilitys[i].getAllDeltaX( is_flat= True ),2 ) )
		
		if dtype == 'hist' :
			return self._util.scheme_hist( data = SquaredX, ylim = ylim, htype = htype, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$\Delta x^2 \; (\mu m)$', density = is_density, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = SquaredX, ylim = ylim, marker = marker, markersize = markersize, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$\Delta x^2 \; (\mu m)$', density = is_density, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )

	def showHistSquaredDifferenceY ( self, dtype = 'hist', ylim = None, is_multi = False, marker = 'o', markersize = 10, is_fit = False, type_fit = 'linear', htype = 'bar', xscale = 'linear', yscale = 'linear', ylabel = 'Count', bins = 10, is_density = False, is_legend = False ):

		SquaredY = []
		for i in range( len(self.motilitys) ):
			SquaredY.append( numpy.power( self.motilitys[i].getAllDeltaY( is_flat= True ),2 ) )
		
		if dtype == 'hist' :
			return self._util.scheme_hist( data = SquaredY, ylim = ylim, htype = htype, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$\Delta y^2 \; (\mu m)$', density = is_density, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = SquaredY, ylim = ylim, marker = marker, markersize = markersize, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$\Delta y^2 \; (\mu m)$', density = is_density, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )

	def showInterQuartileAbsoluteDifferenceXY ( self, is_test = False, is_ns_test = False, showfliers = True, color_box = ['dimgray','lightgray'] ):

		abs_difference_x = []
		abs_difference_y = []
		for i in range( len(self.motilitys) ):
			abs_difference_x.append( numpy.absolute( self.motilitys[i].getAllDeltaX( is_flat= True ) ) )
			abs_difference_y.append( numpy.absolute( self.motilitys[i].getAllDeltaY( is_flat= True ) ) )

		return self._util.scheme_multiple_boxplot( data = [ abs_difference_x, abs_difference_y ], xlabels = self._names , ylabel = r'$ |\Delta x| \; , \; |\Delta y| \; (\mu m)$', spancap = 3, color_box = color_box, is_legend = True, label_legend = [r'$|\Delta x|$',r'$|\Delta y|$'], showfliers = showfliers, is_test = is_test, is_ns_test = is_ns_test )

	def showInterQuartileDifferenceXY ( self, is_test = False, is_ns_test = False, showfliers = True, color_box = ['dimgray','lightgray'] ):

		difference_x = []
		difference_y = []
		for i in range( len(self.motilitys) ):
			difference_x.append( self.motilitys[i].getAllDeltaX( is_flat= True ) )
			difference_y.append( self.motilitys[i].getAllDeltaY( is_flat= True ) )

		return self._util.scheme_multiple_boxplot( data = [ difference_x, difference_y ], xlabels = self._names , ylabel = r'$ \Delta x \; , \; \Delta y \; (\mu m)$', spancap = 3, color_box = color_box, is_legend = True, label_legend = [r'$\Delta x$',r'$\Delta y$'], showfliers = showfliers, is_test = is_test, is_ns_test = is_ns_test )	

	def showInterQuartileDifferencePlusMinusX ( self, is_test = False, is_ns_test = False, showfliers = True, color_box = ['dimgray','lightgray'] ):

		difference_x_plus = []
		difference_x_minus = []
		for i in range( len(self.motilitys) ):
			difference_x = self.motilitys[i].getAllDeltaX( is_flat= True )
			difference_x_plus.append( numpy.absolute( difference_x[ difference_x > 0 ] ) )
			difference_x_minus.append( numpy.absolute( difference_x[ difference_x < 0 ] ) )

		return self._util.scheme_multiple_boxplot( data = [ difference_x_plus, difference_x_minus ], xlabels = self._names , ylabel = r'$\Delta x \; (\mu m)$', spancap = 3, color_box = color_box, is_legend = True, label_legend = [r'$|\Delta x _{+}|$',r'$|\Delta x _{-}|$'], showfliers = showfliers, is_test = is_test, is_ns_test = is_ns_test )	

	def showInterQuartileDifferencePlusMinusY ( self, is_test = False, is_ns_test = False, showfliers = True, color_box = ['dimgray','lightgray'] ):

		difference_y_plus = []
		difference_y_minus = []
		for i in range( len(self.motilitys) ):
			difference_y = self.motilitys[i].getAllDeltaY( is_flat= True )
			difference_y_plus.append( numpy.absolute( difference_y[ difference_y > 0 ] ) )
			difference_y_minus.append( numpy.absolute( difference_y[ difference_y < 0 ] ) )

		return self._util.scheme_multiple_boxplot( data = [ difference_y_plus, difference_y_minus ], xlabels = self._names , ylabel = r'$\Delta y \; (\mu m)$', spancap = 3, color_box = color_box, is_legend = True, label_legend = [r'$|\Delta y _{+}|$',r'$|\Delta y _{-}|$'], showfliers = showfliers, is_test = is_test, is_ns_test = is_ns_test )	

	# noah, joseph, mosses effect

	def showMosesEffectXTimeTracking (self, is_fit = False, type_fit = 'linear', window = (0,10), is_multi = False, is_legend = False, marker = 'o', markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear' ):
		xtime = []
		xpoint = []
		for i in range( len(self.motilitys) ):
			time, point = self.motilitys[i].getAllDeltaX()

			xdata = numpy.nanmedian( numpy.cumsum( numpy.absolute(point), axis = 0), axis = 1 )
			xtime.append( time[1::] )
			xpoint.append( xdata[:-1:] )

		xsize = (6*len(self.motilitys),5) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = xtime, ydata = xpoint, xlabel = 't (min)', ylabel = r'$m[Y_t]$', is_fit = is_fit, type_fit = type_fit, window = window, size = xsize, is_multi = is_multi, is_legend = is_legend, marker = marker, markersize = markersize, alpha = alpha, xscale = xscale , yscale = yscale )
		
	def showNoahEffectXTimeTracking ( self, is_fit = False, type_fit = 'linear', window = (0,10), is_multi = False, is_legend = False, marker = 'o', markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear'):

		xtime = []
		xpoint = []
		for i in range( len(self.motilitys) ):
			time, point = self.motilitys[i].getAllDeltaX()

			xdata = numpy.nanmedian( numpy.cumsum( numpy.power(point,2), axis = 0), axis = 1 )
			xtime.append( time[1::] )
			xpoint.append( xdata[:-1:] )

		xsize = (6*len(xtime),len(xtime)) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = xtime, ydata = xpoint, xlabel = 't (min)', ylabel = r'$m[Z_t]$', is_fit = is_fit, type_fit = type_fit, window = window, size = xsize, is_multi = is_multi, is_legend = is_legend, marker = marker, markersize = markersize, alpha = alpha, xscale = xscale , yscale = yscale )
		
	def showJosephEffectXTimeTracking ( self, is_fit = False, type_fit = 'linear', window = (0,10), is_multi = False, is_legend = False, marker = 'o', markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear' ):

		xtime = []
		xpoint = []
		for i in range( len(self.motilitys) ):
			time, point = self.motilitys[i].getAllDeltaX()
			time = time[1::]
			point = point[:-1:,:]
			Z = numpy.cumsum( numpy.power(point,2), axis = 0)
			X = numpy.cumsum( point, axis = 0)
			R = numpy.full( point.shape , numpy.nan )
			for j in range( point.shape[0] ):
				xtemp = numpy.full( (j+1,point.shape[1]) , numpy.nan )
				for z in range(j+1):
					xtemp[z,:] = X[z,:] - X[j,:]*(z+1)/(j+1)

				R[j,:] = numpy.nanmax( xtemp, axis = 0 ) - numpy.nanmin(xtemp, axis = 0)

			S = numpy.sqrt( (Z.T/time).T - numpy.power( (X.T/time).T ,2) )

			xtime.append( time )
			xpoint.append( numpy.nanmean( R/S , axis = 1 ) )

		xsize = (6*len(xtime),len(xtime)) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = xtime, ydata = xpoint, xlabel = 't (min)', ylabel = r'$E[R_t/S_t]$', is_fit = is_fit, type_fit = type_fit, window = window, size = xsize, is_multi = is_multi, is_legend = is_legend, marker = marker, markersize = markersize, alpha = alpha, xscale = xscale , yscale = yscale )
	
	def showMosesEffectYTimeTracking (self, is_fit = False, type_fit = 'linear', window = (0,10), is_multi = False, is_legend = False, marker = 'o', markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear' ):
		
		xtime = []
		xpoint = []
		for i in range( len(self.motilitys) ):
			time, point = self.motilitys[i].getAllDeltaY()

			xdata = numpy.nanmedian( numpy.cumsum( numpy.absolute(point), axis = 0), axis = 1 )
			xtime.append( time[1::] )
			xpoint.append( xdata[:-1:] )

		xsize = (6*len(xtime),len(xtime)) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = xtime, ydata = xpoint, xlabel = 't (min)', ylabel = r'$m[Y_t]$', is_fit = is_fit, type_fit = type_fit, window = window, size = xsize, is_multi = is_multi, is_legend = is_legend, marker = marker, markersize = markersize, alpha = alpha, xscale = xscale , yscale = yscale )
	
	def showNoahEffectYTimeTracking ( self, is_fit = False, type_fit = 'linear', window = (0,10), is_multi = False, is_legend = False, marker = 'o', markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear'):

		xtime = []
		xpoint = []
		for i in range( len(self.motilitys) ):
			time, point = self.motilitys[i].getAllDeltaY()

			xdata = numpy.nanmedian( numpy.cumsum( numpy.power(point,2), axis = 0), axis = 1 )
			xtime.append( time[1::] )
			xpoint.append( xdata[:-1:] )

		xsize = (6*len(xtime),len(xtime)) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = xtime, ydata = xpoint, xlabel = 't (min)', ylabel = r'$m[Z_t]$', is_fit = is_fit, type_fit = type_fit, window = window, size = xsize, is_multi = is_multi, is_legend = is_legend, marker = marker, markersize = markersize, alpha = alpha, xscale = xscale , yscale = yscale )

	def showJosephEffectYTimeTracking ( self, is_fit = False, type_fit = 'linear', window = (0,10), is_multi = False, is_legend = False, marker = 'o', markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear' ):

		xtime = []
		xpoint = []
		for i in range( len(self.motilitys) ):
			time, point = self.motilitys[i].getAllDeltaY()
			time = time[1::]
			point = point[:-1:,:]
			Z = numpy.cumsum( numpy.power(point,2), axis = 0)
			X = numpy.cumsum( point, axis = 0)
			R = numpy.full( point.shape , numpy.nan )
			for j in range( point.shape[0] ):
				xtemp = numpy.full( (j+1,point.shape[1]) , numpy.nan )
				for z in range(j+1):
					xtemp[z,:] = X[z,:] - X[j,:]*(z+1)/(j+1)

				R[j,:] = numpy.nanmax( xtemp, axis = 0 ) - numpy.nanmin(xtemp, axis = 0)

			S = numpy.sqrt( (Z.T/time).T - numpy.power( (point.T/time).T ,2) )

			xtime.append( time )
			xpoint.append( numpy.nanmean( R/S , axis = 1 ) )

		xsize = (6*len(xtime),len(xtime)) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = xtime, ydata = xpoint, xlabel = 't (min)', ylabel = r'$E[R_t/S_t]$', is_fit = is_fit, type_fit = type_fit, window = window, size = xsize, is_multi = is_multi, is_legend = is_legend, marker = marker, markersize = markersize, alpha = alpha, xscale = xscale , yscale = yscale )
	
	# power spectral density (PSD) of displacement

	def showAllAndAverageXPSD ( self, n_points = 50, is_f = False, fs = None, is_fit = False, window = (0,5), xscale = 'linear', yscale = 'linear' ):

		const_fits = []
		xf = fs
		if not fs :
			xf = [(None,None)]*len(self.motilitys)

		plt.figure( figsize = ( 8*len(self.motilitys), 5 ) )
		for i in range( len(self.motilitys) ):
			freq, psd, avg_psd, std_psd  = self.motilitys[i].getAllAndAvgXPSD( n_points = n_points, is_f = is_f, min_f = xf[i][0] , max_f = xf[i][1] )
			#avg_freq, avg_psd, std_psd = self.motilitys[i].getAverageXPSD( n_points = n_points, is_f = is_f )
			freq = freq[1::]
			psd = psd[1::,:]
			avg_freq = freq
			avg_psd = avg_psd[1::]
			
			plt.subplot(1, len(self.motilitys), i+1)
			plt.title( self._names[i], fontdict = {'fontsize':self._font_size} )
			plt.plot( freq, psd )
			plt.plot( avg_freq, avg_psd, marker = 'o', markeredgecolor = 'black', markersize = 12, linestyle = '', markerfacecolor = self._colors[i], lw = 2 )
			
			if is_fit :
				
				xfit = numpy.log10(avg_freq[window[0]:window[1]])
				yfit = numpy.log10(avg_psd[window[0]:window[1]])
				i_nonnan = numpy.where( ~numpy.isnan(xfit) )[0]
				xfit = xfit[i_nonnan]
				yfit = yfit[i_nonnan]
				i_noninf = numpy.where( ~numpy.isinf(xfit) )[0]
				xfit = xfit[i_noninf]
				yfit = yfit[i_noninf]

				i_nonnan = numpy.where( ~numpy.isnan(yfit) )[0]
				xfit = xfit[i_nonnan]
				yfit = yfit[i_nonnan]
				i_noninf = numpy.where( ~numpy.isinf(yfit) )[0]
				xfit = xfit[i_noninf]
				yfit = yfit[i_noninf]

				xf_psd = linregress( xfit, yfit )

				plt.plot( avg_freq[window[0]:window[1]],  numpy.power(avg_freq[window[0]:window[1]], xf_psd.slope )*math.pow(10, xf_psd.intercept), linestyle = '-', color = 'black', lw = 4 )
				
				const_fits.append( ( xf_psd.slope, math.pow(10, xf_psd.intercept) ) )

			plt.xscale( xscale )
			plt.yscale( yscale )
			plt.xticks( fontsize = self._font_size )
			plt.yticks( fontsize = self._font_size )
			plt.xlabel( 'f (Hz)', fontdict = { 'size' : self._font_size } )
			plt.ylabel( r'PSD, <PSD> $( \mu m^2 / Hz )$', fontdict = { 'size' : self._font_size } )
			plt.grid( linestyle = ':' )

		plt.tight_layout()

		return const_fits, plt

	def showAllAndAverageYPSD ( self, n_points = 50, is_f = False, fs = None, is_fit = False, window = (0,5), xscale = 'linear', yscale = 'linear' ):

		const_fits = []
		xf = fs
		if not fs :
			xf = [(None,None)]*len(self.motilitys)

		plt.figure( figsize = ( 8*len(self.motilitys), 5 ) )
		for i in range( len(self.motilitys) ):
			freq, psd, avg_psd, std_psd = self.motilitys[i].getAllAndAvgYPSD( n_points = n_points, is_f = is_f, min_f = xf[i][0] , max_f = xf[i][1] )
			freq = freq[1::]
			psd = psd[1::,:]
			avg_freq = freq
			avg_psd = avg_psd[1::]
			
			plt.subplot(1, len(self.motilitys), i+1)
			plt.title( self._names[i], fontdict = {'fontsize':self._font_size} )
			plt.plot( freq, psd )
			plt.plot( avg_freq, avg_psd, marker = 'o', markeredgecolor = 'black', markersize = 12, linestyle = '', markerfacecolor = self._colors[i], lw = 2 )
			
			if is_fit :
				
				xfit = numpy.log10(avg_freq[window[0]:window[1]])
				yfit = numpy.log10(avg_psd[window[0]:window[1]])
				i_nonnan = numpy.where( ~numpy.isnan(xfit) )[0]
				xfit = xfit[i_nonnan]
				yfit = yfit[i_nonnan]
				i_noninf = numpy.where( ~numpy.isinf(xfit) )[0]
				xfit = xfit[i_noninf]
				yfit = yfit[i_noninf]

				i_nonnan = numpy.where( ~numpy.isnan(yfit) )[0]
				xfit = xfit[i_nonnan]
				yfit = yfit[i_nonnan]
				i_noninf = numpy.where( ~numpy.isinf(yfit) )[0]
				xfit = xfit[i_noninf]
				yfit = yfit[i_noninf]

				xf_psd = linregress( xfit, yfit )

				plt.plot( avg_freq[window[0]:window[1]],  numpy.power(avg_freq[window[0]:window[1]], xf_psd.slope )*math.pow(10, xf_psd.intercept), linestyle = '-', color = 'black', lw = 4 )
				
				const_fits.append( ( xf_psd.slope, math.pow(10, xf_psd.intercept) ) )

			plt.xscale( xscale )
			plt.yscale( yscale )
			plt.xticks( fontsize = self._font_size )
			plt.yticks( fontsize = self._font_size )
			plt.xlabel( 'f (Hz)', fontdict = { 'size' : self._font_size } )
			plt.ylabel( r'PSD, <PSD> $( \mu m^2 / Hz )$', fontdict = { 'size' : self._font_size } )
			plt.grid( linestyle = ':' )

		plt.tight_layout()

		return const_fits, plt
	
	# characteristics

	def showGlobalDivisionTime ( self, is_test = False, showfliers = True ):

		divtime = []
		for i in range( len(self.motilitys) ):
			divtime.append( self.motilitys[i].getDivisionTime() )

		return self._util.scheme_single_boxplot( data = divtime, ylabel = 'Division time (min)', is_test = is_test, showfliers = showfliers )

	def showGlobalCellSize ( self, is_test = False, showfliers = True ):

		csize = []
		for i in range( len(self.motilitys) ):
			csize.append( self.motilitys[i].getCellSize() )

		return self._util.scheme_single_boxplot( data = csize, ylabel = 'Cell size ($ \mu m$)', is_test = is_test, showfliers = showfliers )

	def showGlobalAspectRatio ( self, is_test = False, showfliers = True ):

		aratio = []
		for i in range( len(self.motilitys) ):
			aratio.append( self.motilitys[i].getAspectRatio() )

		return self._util.scheme_single_boxplot( data = aratio, ylabel = 'Aspect ratio', is_test = is_test, showfliers = showfliers )

	def showGlobalGrowthRate ( self, is_test = False, showfliers = True ):

		grate = []
		for i in range( len(self.motilitys) ):
			grate.append( self.motilitys[i].getGrowthRate() )

		return self._util.scheme_single_boxplot( data = grate, ylabel = 'Growth rate ($\mu m.min^{-1}$)', is_test = is_test, showfliers = showfliers )

	def showGroupDivTimCSizeARGroRate ( self, is_test = False, showfliers = True ):

		divtime = []
		csize = []
		aratio = []
		grate = []
		for i in range( len(self.motilitys) ):
			divtime.append( self.motilitys[i].getDivisionTime() )
			csize.append( self.motilitys[i].getCellSize() )
			aratio.append( self.motilitys[i].getAspectRatio() )
			grate.append( self.motilitys[i].getGrowthRate() )

		xis_test = is_test
		xshowfliers = showfliers
		if type(is_test) is bool :
			xis_test = [is_test]*4

		if type(showfliers) is bool :
			xshowfliers = [showfliers]*4

		plt.figure( figsize = ( 1.5*len(self.motilitys)*4 , 5) )
		plt.subplot(141)
		self._util.scheme_single_boxplot( data = divtime, ylabel = 'Division time (min)', is_sub = True, is_test = xis_test[0], showfliers = xshowfliers[0] )
		plt.subplot(142)
		self._util.scheme_single_boxplot( data = csize, ylabel = 'Cell size ($ \mu m$)', is_sub = True, is_test = xis_test[1], showfliers = xshowfliers[1] )
		plt.subplot(143)
		self._util.scheme_single_boxplot( data = aratio, ylabel = 'Aspect ratio', is_sub = True, is_test = xis_test[2], showfliers = xshowfliers[2] )
		plt.subplot(144)
		self._util.scheme_single_boxplot( data = grate, ylabel = 'Growth rate ($\mu m.min^{-1}$)', is_sub = True, is_test = xis_test[3], showfliers = xshowfliers[3] )
		
		plt.tight_layout()

		return plt

	def showStepEndAspectRatio ( self, is_legend = False, vlines_x = [], is_fit = False, type_fit = 'linear', xscale = 'linear', yscale = 'linear' ):

		steps = []
		aspect_ratio = []
		for i in range( len(self.motilitys) ):
			xsteps = (self.motilitys[i].getGlobalLengthPoints() - 1)
			xaspect = self.motilitys[i].getGlobalEndAspectRatio()
			steps.append( xsteps )
			aspect_ratio.append( xaspect )
			
		return self._util.scheme_scatter( is_multi = True, vlines_x = vlines_x, size = (6*len(self.motilitys),5), xdata = aspect_ratio, ydata = steps, xlabel = r'$Aspect \; ratio ( t = t_{end} )$', ylabel = 'Number of Steps', is_legend = is_legend, xscale = xscale, yscale = yscale, is_fit = is_fit, type_fit = type_fit )
		
	# detail speed

	def showScatterAllXYSpeed ( self, is_color = False, marker = 'o', markersize = 12, alpha = 0.2, is_legend = False ):

		xspeed = []
		yspeed = []
		for i in range( len(self.motilitys) ):
			xsp = self.motilitys[i].getAllVelocityX()
			ysp = self.motilitys[i].getAllVelocityY()
			xspeed.append(xsp)
			yspeed.append(ysp)
		
		return self._util.scheme_scatter( xdata = xspeed, ydata = yspeed, xlabel = r'$\nu_{x} \; (\mu m.min^{-1})$', ylabel = r'$\nu_{y} \; (\mu m.min^{-1})$', size = (6*len(xspeed),len(xspeed)), is_multi = True, is_color = is_color, is_legend = is_legend, is_axis_equal = True, marker = marker, markersize = markersize, alpha = alpha )

	def showHistAllSpeed ( self, is_multi = False, ylim = None, htype = 'bar', dtype = 'hist', is_fit = False, type_fit = 'linear', marker = 'o', markersize = 12, xscale = 'linear', yscale = 'linear', ylabel = 'Count', bins = 10, is_density = False, is_legend = False ):

		speed = []
		for i in range( len(self.motilitys) ):
			speed.append( self.motilitys[i].getAllVelocity( is_flat= True ) )
		
		if dtype == 'hist':
			return self._util.scheme_hist( data = speed, ylim = ylim, ylabel = ylabel, xlabel = r'$\nu \; (\mu m.min^{-1})$', htype = htype, density = is_density, is_fit = is_fit, type_fit = type_fit, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )	
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = speed, ylim = ylim, marker = marker, markersize = markersize, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$\nu \; (\mu m.min^{-1})$', density = is_density, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )

	def showHistAllSquaredSpeed ( self, is_multi = False, ylim = None, htype = 'bar', dtype = 'hist', is_fit = False, type_fit = 'linear', marker = 'o', markersize = 12, xscale = 'linear', yscale = 'linear', ylabel = 'Count', bins = 10, is_density = False, is_legend = False ):

		squared_speed = []
		for i in range( len(self.motilitys) ):
			squared_speed.append( numpy.power( self.motilitys[i].getAllVelocity( is_flat= True ),2 ) )
		
		if dtype == 'hist':
			return self._util.scheme_hist( data = squared_speed, ylim = ylim, ylabel = ylabel, xlabel = r'$\nu^2 \; (\mu m.min^{-1})$', htype = htype, density = is_density, is_fit = True, type_fit = type_fit, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = squared_speed, ylim = ylim, marker = marker, markersize = markersize, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$\nu^2 \; (\mu m.min^{-1})$', density = is_density, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )

	def showAverageNormVelocityAutocorrelation ( self, show_std = False, size = (12,5), xscale = 'linear', yscale = 'linear', is_legend = True ) :
		timeVelAuto = []
		avgVelAuto = []
		stdVelAuto = []
		for i in range( len(self.motilitys) ):
			time, average, std = self.motilitys[i].getAverageNormVelocityCorrelation()		
			timeVelAuto.append( time )
			avgVelAuto.append( average )
			if show_std :
				stdVelAuto.append( std )

		return self._util.scheme_plot_fill( xdata = timeVelAuto, ydata = avgVelAuto, ystd = stdVelAuto, xlabel = 'Time (min)', ylabel = 'Velocity Autocorrelation', is_legend = is_legend, xscale = xscale, yscale = yscale, size = size )

	def showAllAndAveragePowerSpectrumVelocityAutocorrelation ( self, is_fit = False, window = (0,20), bins = 50, xscale = 'linear', yscale = 'linear' ):
		
		def fit_psd(f, slope, const):
			return slope*f + const

		const_fits = []

		plt.figure( figsize = ( 8*len(self.motilitys), 5 ) )
		for i in range( len(self.motilitys) ):
			time, point = self.motilitys[i].getAllVelocityCorrelationPowerSpectrum()
			time = time[1::]
			point = point[1::,:]

			xtime = time.flatten()
			xpsd = point.flatten()
			i_nonnan = numpy.where( ~numpy.isnan(xpsd) )[0]
					
			xtime = xtime[ i_nonnan ]
			xpsd = xpsd[ i_nonnan ]

			ybin, xbin, binnumber = binned_statistic( xtime, xpsd, 'mean', bins = bins )

			plt.subplot(1, len(self.motilitys), i+1)
			plt.plot( time, point )
			plt.plot( xbin[1::], ybin, marker = 'o', markeredgecolor = 'black', markersize = 12, linestyle = '', markerfacecolor = self._colors[i], lw = 2 )
			
			if is_fit :
				
				xfit = numpy.log10(xbin[1::][window[0]:window[1]])
				yfit = numpy.log10( ybin[ window[0]:window[1] ] )
				i_nonnan = numpy.where( ~numpy.isnan(xfit) )[0]
				xfit = xfit[i_nonnan]
				yfit = yfit[i_nonnan]
				i_noninf = numpy.where( ~numpy.isinf(xfit) )[0]
				xfit = xfit[i_noninf]
				yfit = yfit[i_noninf]

				i_nonnan = numpy.where( ~numpy.isnan(yfit) )[0]
				xfit = xfit[i_nonnan]
				yfit = yfit[i_nonnan]
				i_noninf = numpy.where( ~numpy.isinf(yfit) )[0]
				xfit = xfit[i_noninf]
				yfit = yfit[i_noninf]

				xf_psd = curve_fit( fit_psd, xfit, yfit )
				
				plt.plot( xbin[1::][window[0]:window[1]],  numpy.power(xbin[1::][window[0]:window[1]], xf_psd[0][0] )*math.pow(10, xf_psd[0][1]), linestyle = '-', color = 'black', lw = 4 )
				plt.text( numpy.average(xbin[1::][window[1]]), numpy.average(ybin[window[1]]), ' slope '+str( round(xf_psd[0][0],2) ), color = 'black' )

				const_fits.append( ( xf_psd[0][0], math.pow(10, xf_psd[0][1]) ) )

			plt.xscale( xscale )
			plt.yscale( yscale )
			plt.xticks( fontsize = self._font_size )
			plt.yticks( fontsize = self._font_size )
			plt.xlabel( 'f (Hz)', fontdict = { 'size' : self._font_size } )
			plt.ylabel( r'P(f), <P(f)> $( \mu m^2 / Hz )$', fontdict = { 'size' : self._font_size } )
			plt.grid( linestyle = ':' )

		plt.tight_layout()

		return const_fits, plt

	
	# all MSD, TAMSD, MMA, TAMMA
	# scale = {normal, length}
	def showErgodicityBreakingParameterTAMSD ( self, scale = 'normal', is_multi= False, is_color = False, is_legend = False, xscale = 'linear', yscale = 'linear' ) :

		time = []
		eb = []

		for i in range( len(self.motilitys) ):
			xtime, xeb, = self.motilitys[i].getErgodicityBreakingParameterTAMSD()
			if scale == 'normal':
				time.append( xtime )
			else: 
				i_nonnan = numpy.where( ~numpy.isnan(xeb) )[0]
				time.append( xtime/i_nonnan[-1] )

			eb.append( xeb )

		xsize = (6*len(self.motilitys), 5) if is_multi else (6, 5)
		xlabel = r'$\Delta$' if scale == 'normal' else r'$\Delta/T$'
		xinfo_fit, xplt = self._util.scheme_scatter( xdata = time, ydata = eb, xlabel = xlabel, ylabel = 'EB', size = xsize, alpha = 0.5, is_multi = is_multi, is_color = is_color, is_legend = is_legend, xscale = xscale, yscale = yscale )
		if scale == 'normal':
			if not is_multi:
				i_nonnan = numpy.where( ~numpy.isnan( eb[0] ) )[0]
				xplt.plot([0, time[0][i_nonnan[-1]]],[0,0], color= 'black', linestyle = ':')
		else:
			if not is_multi:
				xplt.plot(time[0],(4/3)*time[0], color= 'black', linestyle = ':')

		return xinfo_fit, xplt

	def showTAMSDMSDRatio ( self, is_multi= False, is_color = False, is_legend = False, xscale = 'linear', yscale = 'linear' ) :

		time = []
		eb = []
				
		for i in range( len(self.motilitys) ):
			xtime, xeb, = self.motilitys[i].getTAMSDMSDRatio()	
			time.append( xtime )
			eb.append( xeb )

		xsize = (6*len(self.motilitys), 5) if is_multi else (6, 5)
		xinfo_fit, xplt = self._util.scheme_scatter( xdata = time, ydata = eb, xlabel = r'$\Delta \; (min)$', ylabel = r'$\mathcal{EB}$', size = xsize, alpha = 0.5, is_multi = is_multi, is_color = is_color, is_legend = is_legend, xscale = xscale, yscale = yscale )
		if not is_multi:
			i_nonnan = numpy.where( ~numpy.isnan(eb[0]) )[0]
			xplt.plot( [0, time[0][i_nonnan[-1]] ], [1,1], color = 'black', linestyle = ':' )

		return xinfo_fit, xplt

	def showAverageTAMSD ( self, is_multi = False, window = None, is_fit = False, is_fit_all = False, type_fit = 'linear', xscale = 'linear', yscale = 'linear', is_legend = False, marker = 'o', markersize = 100, alpha = 0.5 ) :
		timemsd = []
		avgmsd = []
		stdmsd = []

		for i in range( len(self.motilitys) ):
			time, average, std = self.motilitys[i].getAverageTAMSD()	
			timemsd.append( time )
			avgmsd.append( average )

		xsize = (6*len(self.motilitys), 5) if is_multi else (6, 5)
		return self._util.scheme_scatter( xdata = timemsd, ydata = avgmsd, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit, window = window, marker = marker, markersize = markersize, xlabel = 'Time (min)', ylabel = r'<TAMSD> $(\mu m^2)$', size = xsize, xscale = xscale, yscale = yscale, is_legend = is_legend, alpha = alpha )
		
	def showMSD ( self, is_multi = False, window = None, is_fit = False, is_fit_all = False, is_legend = False, marker = 'o', markersize = 100, alpha = 0.5, type_fit = 'linear', xscale = 'linear', yscale = 'linear' ) :
		timemsd = []
		avgmsd = []
		stdmsd = []
		
		for i in range( len(self.motilitys) ):
			time, average, std = self.motilitys[i].getMSD()	
			timemsd.append( time )
			avgmsd.append( average )

		xsize = (6*len(self.motilitys), 5) if is_multi else (6, 5)
		return self._util.scheme_scatter( xdata = timemsd, ydata = avgmsd, xlabel = 'Time (min)', ylabel = 'MSD', size = xsize, xscale = xscale, yscale = yscale, is_legend = is_legend, marker = marker, markersize = markersize, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit, window = window, alpha = alpha )
		
	def showXMSD ( self, is_fit = False, type_fit = 'linear', window = (0,10), is_multi = False, is_legend = False, marker = 'o', markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear' ):

		xtime = []
		xmsd = []
		for i in range( len(self.motilitys) ):
			time, mean, std = self.motilitys[i].getXMSD()

			xtime.append( time )
			xmsd.append( mean )

		xsize = (6*len(self.motilitys),5) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = xtime, ydata = xmsd, xlabel = 't (min)', ylabel = r'$MSD_{x}( \mu m^2 )$', is_fit = is_fit, type_fit = type_fit, window = window, size = xsize, is_multi = is_multi, is_legend = is_legend, marker = marker, markersize = markersize, alpha = alpha, xscale = xscale , yscale = yscale )

	def showYMSD ( self, is_fit = False, type_fit = 'linear', window = (0,10), is_multi = False, is_legend = False, marker = 'o', markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear' ):

		xtime = []
		xmsd = []
		for i in range( len(self.motilitys) ):
			time, mean, std = self.motilitys[i].getYMSD()

			xtime.append( time )
			xmsd.append( mean )

		xsize = (6*len(self.motilitys),5) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = xtime, ydata = xmsd, xlabel = 't (min)', ylabel = r'$MSD_{y}( \mu m^2 )$', is_fit = is_fit, type_fit = type_fit, window = window, size = xsize, is_multi = is_multi, is_legend = is_legend, marker = marker, markersize = markersize, alpha = alpha, xscale = xscale , yscale = yscale )

	def showMME ( self, is_fit = False, type_fit = 'linear', window = (0,10), is_multi = False, is_legend = False, marker = 'o', markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear' ) :
		timemme = []
		avgmme = []

		for i in range( len(self.motilitys) ):
			time, average, std = self.motilitys[i].getMME()	
			timemme.append( time )
			avgmme.append( average )
		
		xsize = (6*len(self.motilitys),5) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = timemme, ydata = avgmme, xlabel = 'Time (min)', ylabel = 'MME', size = xsize, xscale = xscale, yscale = yscale, is_legend = is_legend, marker = marker, markersize = markersize, alpha = alpha, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit, window = window )
		
	def showXMME ( self, is_fit = False, type_fit = 'linear', window = (0,10), is_multi = False, is_legend = False, marker = 'o', markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear' ):

		xtime = []
		xmme = []
		for i in range( len(self.motilitys) ):
			time, mean, std = self.motilitys[i].getXMME()

			xtime.append( time )
			xmme.append( mean )

		xsize = (6*len(self.motilitys),5) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = xtime, ydata = xmme, xlabel = 't (min)', ylabel = r'$MME_{x}( \mu m^2 )$', is_fit = is_fit, type_fit = type_fit, window = window, size = xsize, is_multi = is_multi, is_legend = is_legend, marker = marker, markersize = markersize, alpha = alpha, xscale = xscale , yscale = yscale )

	def showYMME ( self, is_fit = False, type_fit = 'linear', window = (0,10), is_multi = False, is_legend = False, marker = 'o', markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear' ):

		xtime = []
		xmme = []
		for i in range( len(self.motilitys) ):
			time, mean, std = self.motilitys[i].getYMME()

			xtime.append( time )
			xmme.append( mean )

		xsize = (6*len(self.motilitys),5) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = xtime, ydata = xmme, xlabel = 't (min)', ylabel = r'$MME_{y}( \mu m^2 )$', is_fit = is_fit, type_fit = type_fit, window = window, size = xsize, is_multi = is_multi, is_legend = is_legend, marker = marker, markersize = markersize, alpha = alpha, xscale = xscale , yscale = yscale )

	def showHistAmplitudeTAMSD ( self, dtype = 'hist', ylim = None, is_multi = False, marker = 'o', markersize = 10, is_fit = False, type_fit = 'normal', htype = 'bar', xscale = 'linear', yscale = 'linear', ylabel = 'Count', bins = 10, is_density = False, is_legend = False ):

		amplitude = []
		for i in range( len(self.motilitys) ):
			amplitude.append( self.motilitys[i].getAmplitudeTAMSD( is_flat_not_nan = True )	)
		
		if dtype == 'hist':
			return self._util.scheme_hist( data = amplitude, ylim = ylim, htype = htype, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$\xi$', density = is_density, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = amplitude, ylim = ylim, marker = marker, markersize = markersize, is_fit = is_fit, ylabel = ylabel, xlabel = r'$\xi$', density = is_density, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )

	def showAllTAMSD ( self, is_color = False, is_legend = False, is_multi = True, xscale = 'linear', yscale = 'linear' ) :

		timemsd = []
		avgmsd = []
		for i in range( len(self.motilitys) ):
			time, average = self.motilitys[i].getAllTAMSD()		
			timemsd.append( time )
			avgmsd.append( average )

		xsize = (6*len(self.motilitys),5) if is_multi else (6,5)
		return self._util.scheme_plot_fill( xdata = timemsd, ydata = avgmsd, ystd = [], xlabel = 'Time (min)', ylabel = 'TAMSD', size = xsize, xscale = xscale, yscale = yscale, is_multi = is_multi, is_color = is_color, is_legend = is_legend )

	def showAverageAndTimeAverageMSD ( self, is_fit = False, window = (0,20), xscale = 'linear', yscale = 'linear' ) :

		def fit_tamsd(t, alpha, org):
			return alpha*t + org

		const_fits = []

		plt.figure( figsize = ( 8*len(self.motilitys), 5 ) )

		for i in range( len(self.motilitys) ):
			time_tamsd, y_tamsd = self.motilitys[i].getAllTAMSD()
			time_etamsd, avg_etamsd, std_etamsd = self.motilitys[i].getAverageTAMSD()
			time_emsd, avg_emsd, std_emsd = self.motilitys[i].getMSD()	

			plt.subplot(1, len(self.motilitys), i+1)

			plt.plot( time_tamsd, y_tamsd )
			plt.plot( time_etamsd, avg_etamsd, marker = 'o', markeredgecolor = self._colors[i], markersize = 12, linestyle = '', markerfacecolor = (1,1,1,0.1), lw = 2 )
			plt.plot( time_emsd, avg_emsd, marker = 's', markeredgecolor = self._colors[i], markersize = 12, linestyle = '', markerfacecolor = (1,1,1,0.1), lw = 2 )
			
			if is_fit :
				self.motilitys[i].setFitTAMSD( window = window )

				xf_etamsd = curve_fit( fit_tamsd, numpy.log10(time_etamsd[window[0]:window[1]]), numpy.log10( avg_etamsd[ window[0]:window[1] ] ) )
				xf_emsd = curve_fit( fit_tamsd, numpy.log10(time_emsd[window[0]:window[1]]), numpy.log10( avg_emsd[ window[0]:window[1] ] ) )

				plt.plot( time_etamsd[window[0]:window[1]],  numpy.power(time_etamsd[window[0]:window[1]], xf_etamsd[0][0] )*math.pow(10, xf_etamsd[0][1]), linestyle = '-', color = self._colors[i], alpha = 0.5, lw = 2 )
				plt.text( numpy.average(time_etamsd[0]), numpy.average(avg_etamsd[0]), ' slope '+str( round(xf_etamsd[0][0],2) ), color = self._colors[i] )

				plt.plot( time_emsd[window[0]:window[1]],  numpy.power(time_emsd[window[0]:window[1]], xf_emsd[0][0] )*math.pow(10, xf_emsd[0][1]), linestyle = '-', color = self._colors[i], alpha = 0.5, lw = 2 )
				plt.text( numpy.average(time_emsd[0]), numpy.average(avg_emsd[0]), ' slope '+str( round(xf_emsd[0][0],2) ), color = self._colors[i] )
				
				const_fits.append( ( math.pow(10, xf_etamsd[0][1]), xf_etamsd[0][0], math.pow(10, xf_emsd[0][1]), xf_emsd[0][0] ) )

			plt.xscale( xscale )
			plt.yscale( yscale )
			plt.xticks( fontsize = self._font_size )
			plt.yticks( fontsize = self._font_size )
			plt.xlabel( 'time (min)', fontdict = { 'size' : self._font_size } )
			plt.ylabel( r'MSD, TAMSD, <TAMSD> $(\mu m^2)$', fontdict = { 'size' : self._font_size } )
			plt.grid( linestyle = ':' )

		plt.tight_layout()

		return const_fits, plt
	
	def showFirstValueTAMSD ( self, is_fit = False, xscale = 'linear', yscale = 'linear' ):

		def fit_linear(x, m, c):
			return x*m + c

		info_fit = []
		plt.figure( figsize = ( 6*len(self.motilitys), 5 ) )
		for i in range( len(self.motilitys) ):
			
			time_tamsd, y_tamsd = self.motilitys[i].getAllTAMSD()
			aspect_ratio = self.motilitys[i].getGlobalEndAspectRatio()
			steps = (self.motilitys[i].getGlobalLengthPoints() - 1)

			plt.subplot(1, len(self.motilitys), i+1)
			plt.scatter( steps,y_tamsd[1,:], marker = 'o', c = self._colors[i], s = aspect_ratio*10 )
			if is_fit:

				xrest = curve_fit( fit_linear, numpy.log10(steps), numpy.log10(y_tamsd[1,:]) )
				plt.plot( steps, numpy.power(steps,xrest[0][0])*math.pow(10,xrest[0][1]), color = 'black', linestyle ='-', lw = 2 )
				plt.text( numpy.average(steps[0]), numpy.average(y_tamsd[1,0]), ' slope \n '+str( round(xrest[0][0],2) ), color = 'black', fontsize = self._font_size )
				info_fit.append( {'slope':xrest[0][0], 'constant':math.pow(10,xrest[0][1])} )
				
			plt.xscale( xscale )
			plt.yscale( yscale )
			plt.xticks( fontsize = self._font_size )
			plt.yticks( fontsize = self._font_size )
			plt.xlabel( 'steps', fontdict = { 'size' : self._font_size } )
			plt.ylabel( r'$TAMSD \; (t=\Delta_0) \; (\mu m^2)$', fontdict = { 'size' : self._font_size } )
			plt.grid( linestyle = ':' )

		plt.tight_layout()

		return info_fit, plt

	def showTimeAverageMSD ( self, is_multi = False, is_fit = False, window = (0,20), alpha = 1, xscale = 'linear', yscale = 'linear' ):

		def fit_tamsd(t, alpha, org):
			return alpha*t + org

		const_fits = []

		if is_multi :
			plt.figure( figsize = ( 8*len(self.motilitys), 5 ) )
		else:
			plt.figure( figsize = ( 8, 5 ) )

		time_tamsd = []
		y_tamsd = []
		time_etamsd = []
		avg_etamsd = []
		std_etamsd = []

		for i in range( len(self.motilitys) ):
			xtime_tamsd, xy_tamsd = self.motilitys[i].getAllTAMSD()
			xtime_etamsd, xavg_etamsd, xstd_etamsd = self.motilitys[i].getAverageTAMSD()

			time_tamsd.append( xtime_tamsd )
			y_tamsd.append( xy_tamsd )
			time_etamsd.append( xtime_etamsd )
			avg_etamsd.append( xavg_etamsd )
			std_etamsd.append( xstd_etamsd )
			
		for i in range( len(self.motilitys) ):

			if is_multi:
				plt.subplot(1, len(self.motilitys), i+1)
				plt.plot( time_tamsd[i], y_tamsd[i], color = self._colors[i], alpha = alpha )
				plt.plot( time_etamsd[i], avg_etamsd[i], marker = 'o', markeredgecolor = 'black', markersize = 12, linestyle = '', markerfacecolor = self._colors[i], lw = 2, label = self._names[i] )
			else:
				plt.plot( time_tamsd[i], y_tamsd[i], color = self._colors[i], alpha = alpha )

			if is_fit and is_multi :
				self.motilitys[i].setFitTAMSD( window = window )

				xf_etamsd = curve_fit( fit_tamsd, numpy.log10(time_etamsd[i][window[0]:window[1]]), numpy.log10( avg_etamsd[i][ window[0]:window[1] ] ) )
				
				plt.plot( time_etamsd[i][window[0]:window[1]],  numpy.power(time_etamsd[i][window[0]:window[1]], xf_etamsd[0][0] )*math.pow(10, xf_etamsd[0][1]), linestyle = ':', color = self._colors[i], lw = 2 )
				plt.text( numpy.average(time_etamsd[i][0]), numpy.average(avg_etamsd[i][0]), ' slope '+str( round(xf_etamsd[0][0],2) ), color = 'black' )

				const_fits.append( { 'intercept':math.pow(10, xf_etamsd[0][1]), 'slope': xf_etamsd[0][0] } )
			if is_multi :
				plt.xscale( xscale )
				plt.yscale( yscale )
				plt.xticks( fontsize = self._font_size )
				plt.yticks( fontsize = self._font_size )
				plt.xlabel( 'time (min)', fontdict = { 'size' : self._font_size } )
				plt.ylabel( r'TAMSD, <TAMSD> $(\mu m^2)$', fontdict = { 'size' : self._font_size } )
				plt.grid( linestyle = ':' )

		if not is_multi :
			xls = ['solid','dotted','dashed','dashdot','loosely dotted','densely dotted','long dash with offset']
			xtext_legend = []
			xlegend = []
			for i in range( len(self.motilitys) ):
				plt.plot( time_etamsd[i], avg_etamsd[i], marker = 'o', markeredgecolor = 'black', markersize = 12, linestyle = '', markerfacecolor = self._colors[i], lw = 2 )
				
				xm_legend = Line2D([0], [0], markerfacecolor = self._colors[i], marker = 'o', markersize = 12, markeredgecolor = 'black', linestyle = '', lw = 2 )
				xlabel = self._names[i]
				if is_fit :
					self.motilitys[i].setFitTAMSD( window = window )

					xf_etamsd = curve_fit( fit_tamsd, numpy.log10(time_etamsd[i][window[0]:window[1]]), numpy.log10( avg_etamsd[i][ window[0]:window[1] ] ) )
					
					plt.plot( time_etamsd[i][window[0]:window[1]],  numpy.power(time_etamsd[i][window[0]:window[1]], xf_etamsd[0][0] )*math.pow(10, xf_etamsd[0][1]), linestyle = xls[i], color = 'black', lw = 2 )
					
					xlabel = xlabel + ' (slope '+str( round(xf_etamsd[0][0],2) )+')'
					xm_legend = Line2D([0], [0], markerfacecolor = self._colors[i], marker = 'o', markersize = 12, markeredgecolor = 'black', linestyle = xls[i], color = 'black', lw = 2 )
					const_fits.append( { 'intercept': math.pow(10, xf_etamsd[0][1]), 'slope': xf_etamsd[0][0] } )
				
				xtext_legend.append( xlabel )
				xlegend.append( xm_legend )


			plt.xscale( xscale )
			plt.yscale( yscale )
			plt.xticks( fontsize = self._font_size )
			plt.yticks( fontsize = self._font_size )
			plt.xlabel( 'time (min)', fontdict = { 'size' : self._font_size } )
			plt.ylabel( r'TAMSD, <TAMSD> $(\mu m^2)$', fontdict = { 'size' : self._font_size } )
			plt.legend( xlegend, xtext_legend, frameon = False )
			plt.grid( linestyle = ':' )
				
		plt.tight_layout()

		return const_fits, plt

	def showMSDAndAverageTAMSD( self, is_fit = False, type_fit = 'linear', window = (0,20), xscale = 'linear', yscale = 'linear' ):

		xtype_fit = [type_fit]*len(self.motilitys) if type(type_fit) is not list else type_fit
		xwindow = [window]*len(self.motilitys) if type(window) is not list else window

		const_fits = []

		plt.figure( figsize = ( 8, 5 ) )

		for i in range( len(self.motilitys) ):
			time_etamsd, avg_etamsd, std_etamsd = self.motilitys[i].getAverageTAMSD()
			time_emsd, avg_emsd, std_emsd = self.motilitys[i].getMSD()

			plt.plot( time_etamsd, avg_etamsd, marker = 'o', markeredgecolor = self._colors[i], markersize = 12, linestyle = '', markerfacecolor = (1,1,1,0), lw = 2 )
			plt.plot( time_emsd, avg_emsd, marker = 's', markeredgecolor = self._colors[i], markersize = 12, linestyle = '', markerfacecolor = (1,1,1,0), lw = 2 )
			
			if is_fit:

				if xtype_fit[i] == 'linear' :

					x_fit_etamsd = time_etamsd[xwindow[i][0]:xwindow[i][1]]
					y_fit_etamsd = avg_etamsd[xwindow[i][0]:xwindow[i][1]]
					i_nonnan = numpy.where( ~numpy.isnan(y_fit_etamsd) )[0]
					x_fit_etamsd = x_fit_etamsd[i_nonnan]
					y_fit_etamsd = y_fit_etamsd[i_nonnan]

					x_fit_emsd = time_emsd[xwindow[i][0]:xwindow[i][1]]
					y_fit_emsd = avg_emsd[xwindow[i][0]:xwindow[i][1]]
					i_nonnan = numpy.where( ~numpy.isnan(y_fit_emsd) )[0]
					x_fit_emsd = x_fit_emsd[i_nonnan]
					y_fit_emsd = y_fit_emsd[i_nonnan]

					xf_etamsd = linregress( numpy.log10(x_fit_etamsd), numpy.log10(y_fit_etamsd) )
					xf_emsd = linregress( numpy.log10(x_fit_emsd), numpy.log10(y_fit_emsd) )

					plt.plot( x_fit_etamsd,  numpy.power(x_fit_etamsd, xf_etamsd.slope )*math.pow(10, xf_etamsd.intercept), linestyle = '-', color = self._colors[i], lw = 2 )
					plt.text( time_etamsd[0], avg_etamsd[0], r'$slope_{<TAMSD>}$ ~ '+str( round(xf_etamsd.slope,2) ), color = self._colors[i] )

					plt.plot( x_fit_emsd,  numpy.power(x_fit_emsd, xf_emsd.slope )*math.pow(10, xf_emsd.intercept), linestyle = ':', color = self._colors[i], lw = 2 )
					plt.text( time_emsd[1], avg_emsd[1], r'$slope_{MSD}$ ~ '+str( round(xf_emsd.slope,2) ), color = self._colors[i] )
					
					const_fits.append( { 'tamsd intercept': math.pow(10, xf_etamsd.intercept), 'tamsd slope': xf_etamsd.slope, 'msd intercept': math.pow(10, xf_emsd.intercept), 'msd slope': xf_emsd.slope } )

				elif xtype_fit[i] == 'bilinear':

					xsplit_etamsd_1 = time_etamsd[ xwindow[i][0][0]:xwindow[i][0][1] ] if xwindow[i] else time_etamsd
					ysplit_etamsd_1 = avg_etamsd[ xwindow[i][0][0]:xwindow[i][0][1] ] if xwindow[i] else avg_etamsd

					xsplit_etamsd_2 = time_etamsd[ xwindow[i][1][0]:xwindow[i][1][1] ] if xwindow[i] else time_etamsd
					ysplit_etamsd_2 = avg_etamsd[ xwindow[i][1][0]:xwindow[i][1][1] ] if xwindow[i] else avg_etamsd
					
					xsplit_emsd_1 = time_emsd[ xwindow[i][0][0]:xwindow[i][0][1] ] if xwindow[i] else time_emsd
					ysplit_emsd_1 = avg_emsd[ xwindow[i][0][0]:xwindow[i][0][1] ] if xwindow[i] else avg_emsd

					xsplit_emsd_2 = time_emsd[ xwindow[i][1][0]:xwindow[i][1][1] ] if xwindow[i] else time_emsd
					ysplit_emsd_2 = avg_emsd[ xwindow[i][1][0]:xwindow[i][1][1] ] if xwindow[i] else avg_emsd

					xsplit_etamsd_1, ysplit_etamsd_1 = self.__UtilClearXY( xsplit_etamsd_1, ysplit_etamsd_1 )
					xsplit_etamsd_2, ysplit_etamsd_2 = self.__UtilClearXY( xsplit_etamsd_2, ysplit_etamsd_2 )
					xsplit_emsd_1, ysplit_emsd_1 = self.__UtilClearXY( xsplit_emsd_1, ysplit_emsd_1 )
					xsplit_emsd_2, ysplit_emsd_2 = self.__UtilClearXY( xsplit_emsd_2, ysplit_emsd_2 )

					res_etamsd_1 = linregress( numpy.log10(xsplit_etamsd_1), numpy.log10(ysplit_etamsd_1) )
					res_etamsd_2 = linregress( numpy.log10(xsplit_etamsd_2), numpy.log10(ysplit_etamsd_2) )

					res_emsd_1 = linregress( numpy.log10(xsplit_emsd_1), numpy.log10(ysplit_emsd_1) )
					res_emsd_2 = linregress( numpy.log10(xsplit_emsd_2), numpy.log10(ysplit_emsd_2) )

					plt.plot( xsplit_etamsd_1,  numpy.power(xsplit_etamsd_1, res_etamsd_1.slope )*math.pow(10, res_etamsd_1.intercept), linestyle = '-', color = self._colors[i], lw = 2 )
					plt.text( xsplit_etamsd_1[0], ysplit_etamsd_1[0], r'$slope_{<TAMSD>}$ ~ '+str( round(res_etamsd_1.slope,2) ), color = self._colors[i] )

					plt.plot( xsplit_etamsd_2,  numpy.power(xsplit_etamsd_2, res_etamsd_2.slope )*math.pow(10, res_etamsd_2.intercept), linestyle = '-', color = self._colors[i], lw = 2 )
					plt.text( xsplit_etamsd_2[0], ysplit_etamsd_2[0], r'$slope_{<TAMSD>}$ ~ '+str( round(res_etamsd_2.slope,2) ), color = self._colors[i] )

					plt.plot( xsplit_emsd_1,  numpy.power(xsplit_emsd_1, res_emsd_1.slope )*math.pow(10, res_emsd_1.intercept), linestyle = ':', color = self._colors[i], lw = 2 )
					plt.text( xsplit_emsd_1[1], ysplit_emsd_1[1], r'$slope_{MSD}$ ~ '+str( round(res_emsd_1.slope,2) ), color = self._colors[i] )

					plt.plot( xsplit_emsd_2,  numpy.power(xsplit_emsd_2, res_emsd_2.slope )*math.pow(10, res_emsd_2.intercept), linestyle = ':', color = self._colors[i], lw = 2 )
					plt.text( xsplit_emsd_2[1], ysplit_emsd_2[1], r'$slope_{MSD}$ ~ '+str( round(res_emsd_2.slope,2) ), color = self._colors[i] )

					const_fits.append( { 'tamsd slope 1' : res_etamsd_1.slope, 'tamsd intercept 1' : res_etamsd_1.intercept, 'tamsd slope 2' : res_etamsd_2.slope, 'tamsd intercept 2' : res_etamsd_2.intercept } )
					const_fits.append( { 'msd slope 1' : res_emsd_1.slope, 'msd intercept 1' : res_emsd_1.intercept, 'msd slope 2' : res_emsd_2.slope, 'msd intercept 2' : res_emsd_2.intercept } )


		plt.xscale( xscale )
		plt.yscale( yscale )
		plt.xticks( fontsize = self._font_size )
		plt.yticks( fontsize = self._font_size )
		plt.xlabel( 'time (min)', fontdict = { 'size' : self._font_size } )
		plt.ylabel( r'MSD, <TAMSD> $(\mu m^2)$', fontdict = { 'size' : self._font_size } )
		plt.grid( linestyle = ':' )

		return const_fits, plt

	def showAverageAndTimeAverageMME ( self, is_fit = False, window = (0,20), xscale = 'linear', yscale = 'linear' ):

		def fit_tamme(t, alpha, org):
			return alpha*t + org

		const_fits = []

		plt.figure( figsize = ( 8*len(self.motilitys), 5 ) )

		for i in range( len(self.motilitys) ):
			time_tamme, y_tamme = self.motilitys[i].getAllTAMME()
			time_etamme, avg_etamme, std_etamme = self.motilitys[i].getAverageTAMME()
			time_emme, avg_emme, std_emme = self.motilitys[i].getMME()

			plt.subplot(1, len(self.motilitys), i+1)
			plt.plot( time_tamme, y_tamme )
			plt.plot( time_etamme, avg_etamme, marker = 'o', markeredgecolor = self._colors[i], markersize = 12, linestyle = '', markerfacecolor = (1,1,1,0.1), lw = 2 )
			plt.plot( time_emme, avg_emme, marker = 's', markeredgecolor = self._colors[i], markersize = 12, linestyle = '', markerfacecolor = (1,1,1,0.1), lw = 2 )
			
			if is_fit :
				self.motilitys[i].setFitTAMME( window = window )

				xf_etamme = curve_fit( fit_tamme, numpy.log10(time_etamme[window[0]:window[1]]), numpy.log10( avg_etamme[ window[0]:window[1] ] ) )
				xf_emme = curve_fit( fit_tamme, numpy.log10(time_emme[window[0]:window[1]]), numpy.log10( avg_emme[ window[0]:window[1] ] ) )

				plt.plot( time_etamme[window[0]:window[1]],  numpy.power(time_etamme[window[0]:window[1]], xf_etamme[0][0] )*math.pow(10, xf_etamme[0][1]), linestyle = '-', color = self._colors[i], alpha = 0.5, lw = 2 )
				plt.text( numpy.average(time_etamme[0]), numpy.average(avg_etamme[0]), ' slope '+str( round(xf_etamme[0][0],2) ), color = self._colors[i] )

				plt.plot( time_emme[window[0]:window[1]],  numpy.power(time_emme[window[0]:window[1]], xf_emme[0][0] )*math.pow(10, xf_emme[0][1]), linestyle = '-', color = self._colors[i], alpha = 0.5, lw = 2 )
				plt.text( numpy.average(time_emme[0]), numpy.average(avg_emme[0]), ' slope '+str( round(xf_emme[0][0],2) ), color = self._colors[i] )
				
				const_fits.append( ( math.pow(10, xf_etamme[0][1]), xf_etamme[0][0], math.pow(10, xf_emme[0][1]), xf_emme[0][0] ) )

			plt.xscale( xscale )
			plt.yscale( yscale )
			plt.xticks( fontsize = self._font_size )
			plt.yticks( fontsize = self._font_size )
			plt.xlabel( 'time (min)', fontdict = { 'size' : self._font_size } )
			plt.ylabel( r'MME, TAMME, <TAMME> $(\mu m^2)$', fontdict = { 'size' : self._font_size } )
			plt.grid( linestyle = ':' )

		plt.tight_layout()

		return const_fits, plt

	def showTimeAverageMME ( self, is_multi = False, is_fit = False, window = (0,20), xscale = 'linear', yscale = 'linear' ):

		def fit_tamme(t, alpha, org):
			return alpha*t + org

		time_tamme = []
		y_tamme = []
		time_etamme = []
		avg_etamme = []
		std_etamme = []
		for i in range( len(self.motilitys) ):
			xtime_tamme, xy_tamme = self.motilitys[i].getAllTAMME()
			xtime_etamme, xavg_etamme, xstd_etamme = self.motilitys[i].getAverageTAMME()
			time_tamme.append( xtime_tamme )
			y_tamme.append( xy_tamme )
			time_etamme.append( xtime_etamme )
			avg_etamme.append( xavg_etamme )
			std_etamme.append( xstd_etamme )

		const_fits = []

		if is_multi:
			plt.figure( figsize = ( 8*len(self.motilitys), 5 ) )
		else:
			plt.figure( figsize = ( 8, 5 ) )

		for i in range( len(self.motilitys) ):
			
			if is_multi:
				plt.subplot(1, len(self.motilitys), i+1)
				plt.plot( time_tamme[i], y_tamme[i], color = self._colors[i] )
				plt.plot( time_etamme[i], avg_etamme[i], marker = 'o', markeredgecolor = 'black', markersize = 12, linestyle = '', markerfacecolor = self._colors[i], lw = 2 )
			else:
				plt.plot( time_tamme[i], y_tamme[i], color = self._colors[i] )

			if is_fit and is_multi :
				self.motilitys[i].setFitTAMME( window = window )

				xf_etamme = curve_fit( fit_tamme, numpy.log10(time_etamme[i][window[0]:window[1]]), numpy.log10( avg_etamme[i][ window[0]:window[1] ] ) )
				xf_emme = curve_fit( fit_tamme, numpy.log10(time_emme[i][window[0]:window[1]]), numpy.log10( avg_emme[i][ window[0]:window[1] ] ) )

				plt.plot( time_etamme[i][window[0]:window[1]],  numpy.power(time_etamme[i][window[0]:window[1]], xf_etamme[0][0] )*math.pow(10, xf_etamme[0][1]), linestyle = ':', color = self._colors[i], lw = 2 )
				plt.text( numpy.average(time_etamme[i][0]), numpy.average(avg_etamme[i][0]), ' slope '+str( round(xf_etamme[0][0],2) ), color = self._colors[i] )

				const_fits.append( ( math.pow(10, xf_etamme[0][1]), xf_etamme[0][0] ) )

			if is_multi:
				plt.xscale( xscale )
				plt.yscale( yscale )
				plt.xticks( fontsize = self._font_size )
				plt.yticks( fontsize = self._font_size )
				plt.xlabel( 'time (min)', fontdict = { 'size' : self._font_size } )
				plt.ylabel( r'TAMME, <TAMME> $(\mu m^2)$', fontdict = { 'size' : self._font_size } )
				plt.grid( linestyle = ':' )

		if not is_multi:
			for i in range( len(self.motilitys) ):
				plt.plot( time_etamme[i], avg_etamme[i], marker = 'o', markeredgecolor = 'black', markersize = 12, linestyle = '', markerfacecolor = self._colors[i], lw = 2, label = self._names[i] )

				if is_fit :
					self.motilitys[i].setFitTAMME( window = window )

					xf_etamme = curve_fit( fit_tamme, numpy.log10(time_etamme[i][window[0]:window[1]]), numpy.log10( avg_etamme[i][ window[0]:window[1] ] ) )
					xf_emme = curve_fit( fit_tamme, numpy.log10(time_emme[i][window[0]:window[1]]), numpy.log10( avg_emme[i][ window[0]:window[1] ] ) )

					plt.plot( time_etamme[i][window[0]:window[1]],  numpy.power(time_etamme[i][window[0]:window[1]], xf_etamme[0][0] )*math.pow(10, xf_etamme[0][1]), linestyle = ':', color = self._colors[i], lw = 2 )
					plt.text( numpy.average(time_etamme[i][0]), numpy.average(avg_etamme[i][0]), ' slope '+str( round(xf_etamme[0][0],2) ), color = self._colors[i] )

					const_fits.append( ( math.pow(10, xf_etamme[0][1]), xf_etamme[0][0] ) )

			plt.xscale( xscale )
			plt.yscale( yscale )
			plt.xticks( fontsize = self._font_size )
			plt.yticks( fontsize = self._font_size )
			plt.xlabel( 'time (min)', fontdict = { 'size' : self._font_size } )
			plt.ylabel( r'TAMME, <TAMME> $(\mu m^2)$', fontdict = { 'size' : self._font_size } )
			plt.grid( linestyle = ':' )
			plt.legend( frameon = False )

		plt.tight_layout()

		return const_fits, plt

	def showMMEAndAverageTAMME ( self, is_fit = False, type_fit = 'linear', window = (0,20), xscale = 'linear', yscale = 'linear' ):

		xtype_fit = [type_fit]*len(self.motilitys) if type(type_fit) is not list else type_fit
		xwindow = [window]*len(self.motilitys) if type(window) is not list else window

		const_fits = []

		plt.figure( figsize = ( 8, 5 ) )

		for i in range( len(self.motilitys) ):
			time_etamme, avg_etamme, std_etamme = self.motilitys[i].getAverageTAMME()
			time_emme, avg_emme, std_emme = self.motilitys[i].getMME()	

			plt.plot( time_etamme, avg_etamme, marker = 'o', markeredgecolor = self._colors[i], markersize = 12, linestyle = '', markerfacecolor = (1,1,1,0), lw = 2 )
			plt.plot( time_emme, avg_emme, marker = 's', markeredgecolor = self._colors[i], markersize = 12, linestyle = '', markerfacecolor = (1,1,1,0), lw = 2 )
			
			if is_fit:

				if xtype_fit[i] == 'linear':

					x_fit_etamme = time_etamme[xwindow[i][0]:xwindow[i][1]]
					y_fit_etamme = avg_etamme[xwindow[i][0]:xwindow[i][1]]
					
					x_fit_emme = time_emme[xwindow[i][0]:xwindow[i][1]]
					y_fit_emme = avg_emme[xwindow[i][0]:xwindow[i][1]]

					x_fit_etamme, y_fit_etamme = self.__UtilClearXY( x_fit_etamme, y_fit_etamme )
					x_fit_emme, y_fit_emme = self.__UtilClearXY( x_fit_emme, y_fit_emme )
					
					xf_etamme = linregress( numpy.log10(x_fit_etamme), numpy.log10(y_fit_etamme) )
					xf_emme = linregress( numpy.log10(x_fit_emme), numpy.log10(y_fit_emme) )

					plt.plot( x_fit_etamme,  numpy.power(x_fit_etamme, xf_etamme.slope )*math.pow(10, xf_etamme.intercept), linestyle = '-', color = self._colors[i], lw = 2 )
					plt.text( x_fit_etamme[0], y_fit_etamme[0], r'$slope_{<TAMME>}$ ~ '+str( round(xf_etamme.slope,2) ), color = self._colors[i] )

					plt.plot( x_fit_emme,  numpy.power(x_fit_emme, xf_emme.slope )*math.pow(10, xf_emme.intercept), linestyle = ':', color = self._colors[i], lw = 2 )
					plt.text( x_fit_emme[1], y_fit_emme[1], r'$slope_{MME}$ ~ '+str( round(xf_emme.slope,2) ), color = self._colors[i] )
					
					const_fits.append( { 'tamme intercept': math.pow(10, xf_etamme.intercept), 'tamme slope': xf_etamme.slope, 'mme intercept': math.pow(10, xf_emme.intercept), 'mme slope': xf_emme.slope } )

				elif xtype_fit[i] == 'bilinear':

					xsplit_etamme_1 = time_etamme[ xwindow[i][0][0]:xwindow[i][0][1] ] if xwindow[i] else time_etamme
					ysplit_etamme_1 = avg_etamme[ xwindow[i][0][0]:xwindow[i][0][1] ] if xwindow[i] else avg_etamme

					xsplit_etamme_2 = time_etamme[ xwindow[i][1][0]:xwindow[i][1][1] ] if xwindow[i] else time_etamme
					ysplit_etamme_2 = avg_etamme[ xwindow[i][1][0]:xwindow[i][1][1] ] if xwindow[i] else avg_etamme
					
					xsplit_emme_1 = time_emme[ xwindow[i][0][0]:xwindow[i][0][1] ] if xwindow[i] else time_emme
					ysplit_emme_1 = avg_emme[ xwindow[i][0][0]:xwindow[i][0][1] ] if xwindow[i] else avg_emme

					xsplit_emme_2 = time_emme[ xwindow[i][1][0]:xwindow[i][1][1] ] if xwindow[i] else time_emme
					ysplit_emme_2 = avg_emme[ xwindow[i][1][0]:xwindow[i][1][1] ] if xwindow[i] else avg_emme

					xsplit_etamme_1, ysplit_etamme_1 = self.__UtilClearXY( xsplit_etamme_1, ysplit_etamme_1 )
					xsplit_etamme_2, ysplit_etamme_2 = self.__UtilClearXY( xsplit_etamme_2, ysplit_etamme_2 )
					xsplit_emme_1, ysplit_emme_1 = self.__UtilClearXY( xsplit_emme_1, ysplit_emme_1 )
					xsplit_emme_2, ysplit_emme_2 = self.__UtilClearXY( xsplit_emme_2, ysplit_emme_2 )

					res_etamme_1 = linregress( numpy.log10(xsplit_etamme_1), numpy.log10(ysplit_etamme_1) )
					res_etamme_2 = linregress( numpy.log10(xsplit_etamme_2), numpy.log10(ysplit_etamme_2) )

					res_emme_1 = linregress( numpy.log10(xsplit_emme_1), numpy.log10(ysplit_emme_1) )
					res_emme_2 = linregress( numpy.log10(xsplit_emme_2), numpy.log10(ysplit_emme_2) )

					plt.plot( xsplit_etamme_1,  numpy.power(xsplit_etamme_1, res_etamme_1.slope )*math.pow(10, res_etamme_1.intercept), linestyle = '-', color = self._colors[i], lw = 2 )
					plt.text( xsplit_etamme_1[0], ysplit_etamme_1[0], r'$slope_{<TAMME>}$ ~ '+str( round(res_etamme_1.slope,2) ), color = self._colors[i] )

					plt.plot( xsplit_etamme_2,  numpy.power(xsplit_etamme_2, res_etamme_2.slope )*math.pow(10, res_etamme_2.intercept), linestyle = '-', color = self._colors[i], lw = 2 )
					plt.text( xsplit_etamme_2[0], ysplit_etamme_2[0], r'$slope_{<TAMME>}$ ~ '+str( round(res_etamme_2.slope,2) ), color = self._colors[i] )

					plt.plot( xsplit_emme_1,  numpy.power(xsplit_emme_1, res_emme_1.slope )*math.pow(10, res_emme_1.intercept), linestyle = ':', color = self._colors[i], lw = 2 )
					plt.text( xsplit_emme_1[1], ysplit_emme_1[1], r'$slope_{MME}$ ~ '+str( round(res_emme_1.slope,2) ), color = self._colors[i] )

					plt.plot( xsplit_emme_2,  numpy.power(xsplit_emme_2, res_emme_2.slope )*math.pow(10, res_emme_2.intercept), linestyle = ':', color = self._colors[i], lw = 2 )
					plt.text( xsplit_emme_2[1], ysplit_emme_2[1], r'$slope_{MME}$ ~ '+str( round(res_emme_2.slope,2) ), color = self._colors[i] )

					const_fits.append( { 'tamme slope 1' : res_etamme_1.slope, 'tamme intercept 1' : res_etamme_1.intercept, 'tamme slope 2' : res_etamme_2.slope, 'tamme intercept 2' : res_etamme_2.intercept } )
					const_fits.append( { 'mme slope 1' : res_emme_1.slope, 'mme intercept 1' : res_emme_1.intercept, 'mme slope 2' : res_emme_2.slope, 'mme intercept 2' : res_emme_2.intercept } )



		plt.xscale( xscale )
		plt.yscale( yscale )
		plt.xticks( fontsize = self._font_size )
		plt.yticks( fontsize = self._font_size )
		plt.xlabel( 'time (min)', fontdict = { 'size' : self._font_size } )
		plt.ylabel( r'MME, <TAMME> $(\mu m^2)$', fontdict = { 'size' : self._font_size } )
		plt.grid( linestyle = ':' )

		return const_fits, plt

	# all scaling exponent and diffusion coefficient

	def showDiffusionCoefficientScalingExponentTAMSD ( self, is_set_fit_TAMSD = False, window_TAMSD = (1,6), ylim = None, is_multi = False, is_fit = False, is_fit_all = False, type_fit = 'linear', window = None, markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear', is_legend = False ):

		if is_set_fit_TAMSD :
			for i in range( len(self.motilitys) ):
				self.motilitys[i].setFitTAMSD( window = window_TAMSD )
		
		scaling_fit_values = []
		diffussion_fit_values = []
		for i in range( len(self.motilitys) ):
			scaling_fit_values.append( self.motilitys[i].getScalingExponentFit() )
			diffussion_fit_values.append( self.motilitys[i].getGeneralisedDiffusionCoefficientFit() )

		xsize = ( 6*len(self.motilitys) ,5) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = scaling_fit_values, ydata = diffussion_fit_values, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit, window = window, xlabel = r'$ \beta $' , ylabel = r'$ K_{\beta} $', is_multi = is_multi, size = xsize, markersize = markersize, alpha = alpha, xscale = xscale, yscale = yscale, is_legend = is_legend )

	def showDiffusionCoefficientScalingExponentTAMME ( self, is_set_fit_TAMME = False, window_TAMME = (1,6), ylim = None, is_multi = False, is_fit = False, is_fit_all = False, type_fit = 'linear', window = None, markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear', is_legend = False ):

		if is_set_fit_TAMME :
			for i in range( len(self.motilitys) ):
				self.motilitys[i].setFitTAMME( window = window_TAMME )

		scaling_fit_values = []
		diffussion_fit_values = []
		for i in range( len(self.motilitys) ):
			scaling_fit_values.append( self.motilitys[i].getScalingExponentFitTAMME() )
			diffussion_fit_values.append( self.motilitys[i].getGeneralisedDiffusionCoefficientFitTAMME() )

		xsize = ( 6*len(self.motilitys) ,5) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = scaling_fit_values, ydata = diffussion_fit_values, is_fit = is_fit, is_fit_all = is_fit_all, ylim = ylim, type_fit = type_fit, window = window, xlabel = r'$ \beta_{TAMME} $' , ylabel = r'$ K_{\beta}^{TAMME} $', is_multi = is_multi, size = xsize, markersize = markersize, alpha = alpha, xscale = xscale, yscale = yscale, is_legend = is_legend )

	def showHistScalingExponent ( self, ylim = None, is_set_fit_TAMSD = False, window = (1,6), dtype = 'hist', is_multi = False, marker = 'o', markersize = 10, is_fit = False, type_fit = 'normal', htype = 'bar', xscale = 'linear', yscale = 'linear', bins = 10, is_density = False, is_legend = False ):
		
		if is_set_fit_TAMSD :
			for i in range( len(self.motilitys) ):
				self.motilitys[i].setFitTAMSD( window = window )

		scaling_fit_values = []
		for i in range( len(self.motilitys) ):
			scaling_fit_values.append( self.motilitys[i].getScalingExponentFit( is_not_nan = True ) )
		
		y_label = r'$P( \beta )$' if is_density else r'$N( \beta )$'
		if dtype == 'hist' :
			return self._util.scheme_hist( data = scaling_fit_values, ylim = ylim, htype = htype, is_fit = is_fit, type_fit = type_fit, ylabel = y_label, xlabel = r'$\beta$', density = is_density, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = scaling_fit_values, ylim = ylim, marker = marker, markersize = markersize, is_fit = is_fit, ylabel = y_label, xlabel = r'$\beta$', density = is_density, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
	
	def showInterQuartileScalingExponent ( self, is_set_fit_TAMSD = False, window = (1,6), is_test = False, showfliers = True ):

		if is_set_fit_TAMSD :
			for i in range( len(self.motilitys) ):
				self.motilitys[i].setFitTAMSD( window = window )

		scaling_fit_values = []
		for i in range( len(self.motilitys) ):
			scaling_fit_values.append( self.motilitys[i].getScalingExponentFit( is_not_nan = True ) )

		return self._util.scheme_single_boxplot( data = scaling_fit_values, ylabel = r'$\beta$', is_test = is_test, showfliers = showfliers )

	def showHistGeneralizedDiffusionCoefficient ( self, is_norm = False, ylim = None, is_set_fit_TAMSD = False, window = (1,6), type_fit = 'powerlaw', dtype = 'hist', is_multi = False, marker = 'o', markersize = 10, is_fit = False, htype = 'bar', xscale = 'linear', yscale = 'linear', bins = 10, is_density = False, is_legend = False ):
		
		if is_set_fit_TAMSD :
			for i in range( len(self.motilitys) ):
				self.motilitys[i].setFitTAMSD( window = window )

		diffusion_fit_values = []
		for i in range( len(self.motilitys) ):
			xdata = self.motilitys[i].getGeneralisedDiffusionCoefficientFit( is_not_nan = True )
			diffusion_fit_values.append( xdata/numpy.average(xdata) if is_norm else xdata )
		
		x_label = r'$\frac{ K_{\beta} }{ \langle K_{\beta} \rangle }$' if is_norm else r'$K_{\beta}$'
		y_label = ( r'$P( \frac{ K_{\beta} }{ \langle K_{\beta} \rangle } )$' if is_norm else r'$P( K_{\beta} )$' ) if is_density else ( r'$N( \frac{ K_{\beta} }{ \langle K_{\beta} \rangle } )$' if is_norm else r'$N( K_{\beta} )$' )
		if dtype == 'hist' :
			return self._util.scheme_hist( data = diffusion_fit_values, ylim = ylim, htype = htype, is_fit = is_fit, type_fit = type_fit, ylabel = y_label, xlabel = x_label, density = is_density, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = diffusion_fit_values, ylim = ylim, marker = marker, markersize = markersize, is_fit = is_fit, ylabel = y_label, xlabel = x_label, density = is_density, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )

	def showInterQuartileGeneralizedDiffusionCoefficient ( self, is_set_fit_TAMSD = False, window = (1,6), is_test = False, showfliers = True ):

		if is_set_fit_TAMSD :
			for i in range( len(self.motilitys) ):
				self.motilitys[i].setFitTAMSD( window = window )

		diffusion_fit_values = []
		for i in range( len(self.motilitys) ):
			diffusion_fit_values.append( self.motilitys[i].getGeneralisedDiffusionCoefficientFit( is_not_nan = True ) )

		return self._util.scheme_single_boxplot( data = diffusion_fit_values, ylabel = r'$K_{\beta}$', is_test = is_test, showfliers = showfliers )

	def showHistFirstGeneralizedDiffusionCoefficient ( self, ylim = None, is_set_fit_TAMSD = False, window = (1,6), dtype = 'hist', is_multi = False, marker = 'o', markersize = 10, is_fit = False, type_fit = 'normal', htype = 'bar', xscale = 'linear', yscale = 'linear', bins = 10, is_density = False, is_legend = False ):

		if is_set_fit_TAMSD :
			for i in range( len(self.motilitys) ):
				self.motilitys[i].setFitTAMSD( window = window )

		first_diffusion_fit_values = []
		for i in range( len(self.motilitys) ):
			first_diffusion_fit_values.append( self.motilitys[i].getFirstGeneralisedDiffusionCoefficientFit( ) )
		
		y_label = r'$P( K_1 )$' if is_density else r'$N( K_1 )$'
		if dtype == 'hist' :
			return self._util.scheme_hist( data = first_diffusion_fit_values, ylim = ylim, htype = htype, is_fit = is_fit, type_fit = type_fit, ylabel = y_label, xlabel = r'$K_1$', density = is_density, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = first_diffusion_fit_values, ylim = ylim, marker = marker, markersize = markersize, is_fit = is_fit, ylabel = y_label, xlabel = r'$K_1$', density = is_density, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )

	def showInterQuartileFirstGeneralizedDiffusionCoefficient ( self, is_set_fit_TAMSD = False, window = (1,6), is_test = False, showfliers = True ):

		if is_set_fit_TAMSD :
			for i in range( len(self.motilitys) ):
				self.motilitys[i].setFitTAMSD( window = window )

		first_diffusion_fit_values = []
		for i in range( len(self.motilitys) ):
			first_diffusion_fit_values.append( self.motilitys[i].getFirstGeneralisedDiffusionCoefficientFit( is_not_nan = True ) )

		return self._util.scheme_single_boxplot( data = first_diffusion_fit_values, ylabel = r'$K_1$', is_test = is_test, showfliers = showfliers )

	def showCompareScalingExponentTAMSD_TAMME ( self, is_set_fit_TAMSD_TAMME = False, window_TAMSD = (1,6), is_multi = True, is_fit = False, type_fit = 'linear', window = None, alpha = 0.5, xscale = 'linear', yscale = 'linear' ):

		scaling_exponent_tamsd = []
		scaling_exponent_tamme = []
		for i in range( len(self.motilitys) ):
			
			if is_set_fit_TAMSD_TAMME :
				self.motilitys[i].setFitTAMSD( window = window_TAMSD )
				self.motilitys[i].setFitTAMME( window = window_TAMSD )

			scaling_exponent_tamsd.append( self.motilitys[i].getScalingExponentFit() )
			scaling_exponent_tamme.append( self.motilitys[i].getScalingExponentFitTAMME() )
		
		xsize = ( 6*len(self.motilitys), 5 ) if is_multi else ( 6, 5 )

		xinfo_fit, xplt = self._util.scheme_scatter( xdata = scaling_exponent_tamme, ydata = scaling_exponent_tamsd, is_multi = is_multi, is_fit = is_fit, type_fit = type_fit, window = window, markersize = 100, size = xsize, alpha = alpha, xlabel = r'$\beta_{TAMME}$', ylabel = r'$\beta_{TAMSD}$', xscale = xscale, yscale = xscale )
		if not is_multi:
			xplt.plot([0,2],[0,2], color='black', lw = 2, linestyle = ':')

		return xinfo_fit, xplt

	def showErrorCompareScalingExponentTAMSD_TAMME ( self, is_split = False, is_set_fit_TAMSD_TAMME = False, window_TAMSD = (1,6), showfliers = True, is_test = False ) :

		error_scaling_exponent = []
		for i in range( len(self.motilitys) ):
			
			if is_set_fit_TAMSD_TAMME :
				self.motilitys[i].setFitTAMSD( window = window_TAMSD )
				self.motilitys[i].setFitTAMME( window = window_TAMSD )

			xtamsd = self.motilitys[i].getScalingExponentFit()
			xtamme = self.motilitys[i].getScalingExponentFitTAMME()

			xdata = numpy.absolute( xtamme - xtamsd )*100/numpy.absolute(xtamsd)
			if is_split :

				sub = numpy.where( xtamsd < 1 )[0]
				sup = numpy.where( xtamsd > 1 )[0]

				error_scaling_exponent.append( [ xdata[sub], xdata[sup] ] )

			else:
				error_scaling_exponent.append( xdata )
		
		if is_split :
			return self._util.scheme_multiple_boxplot( data = error_scaling_exponent, xlabels = ['Subdiffusion','Superdiffusion'], ylabel = r'$\frac {| \beta_{TAMME} - \beta_{TAMSD} |} {\beta_{TAMSD}} $ (%)', is_legend = True, spancap = 3, showfliers = showfliers, is_test = is_test )
		else:
			return self._util.scheme_single_boxplot( data = error_scaling_exponent, ylabel = r'$\frac {| \beta_{TAMME} - \beta_{TAMSD} |} {\beta_{TAMSD}} $ (%)', showfliers = showfliers, is_test = is_test )
			
	def showCompareGeneralizedDiffusionCoefficientTAMSD_TAMME ( self, is_set_fit_TAMSD_TAMME = False, window_TAMSD = (1,6), is_multi = True, is_fit = False, type_fit = 'linear', window = None, alpha = 0.5, xscale = 'linear', yscale = 'linear' ):

		diffusion_coefficient_tamsd = []
		diffusion_coefficient_tamme = []
		max_dc_tamsd = []
		for i in range( len(self.motilitys) ):
			
			if is_set_fit_TAMSD_TAMME :
				self.motilitys[i].setFitTAMSD( window = window_TAMSD )
				self.motilitys[i].setFitTAMME( window = window_TAMSD )

			diffusion_coefficient_tamsd.append( self.motilitys[i].getGeneralisedDiffusionCoefficientFit() )
			diffusion_coefficient_tamme.append( self.motilitys[i].getGeneralisedDiffusionCoefficientFitTAMME() )
		
		for i in range( len(self.motilitys) ):
			max_dc_tamsd.append( numpy.nanmax( diffusion_coefficient_tamsd[i] ) )

		xsize = ( 6*len(self.motilitys), 5 ) if is_multi else ( 6, 5 )

		xinfo_fit, xplt = self._util.scheme_scatter( xdata = diffusion_coefficient_tamme, ydata = diffusion_coefficient_tamsd, is_multi = is_multi, is_fit = is_fit, type_fit = type_fit, window = window, markersize = 100, size = xsize, alpha = alpha, xlabel = r'$K_{\beta}^{TAMME}$', ylabel = r'$K_{\beta}^{TAMSD}$', xscale = xscale, yscale = xscale )
		if not is_multi:
			xplt.plot([0, numpy.nanmax(max_dc_tamsd) ],[0,numpy.nanmax(max_dc_tamsd)], color='black', lw = 2, linestyle = ':')

		return xinfo_fit, xplt

	# all scaling exponent and diffusion coefficient with aging
	# @ type_time = ageing, norm, init, middle
	def showFirstDiffusionCoeffientTimeExperiment ( self, type_time = 'ageing', is_binned = False, bins = 20, is_set_fit_TAMSD = False, window_TAMSD = (0,6), is_fit = False, type_fit = 'linear', window = None, is_multi = False, is_legend = False, marker = 'o', markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear' ):
		if is_set_fit_TAMSD :
			for i in range( len(self.motilitys) ):
				self.motilitys[i].setFitTAMSD( window = window_TAMSD )

		start_time = []
		first_diffusion_fit_values = []
		for i in range( len(self.motilitys) ):
			first_diffusion_fit_values.append( self.motilitys[i].getFirstGeneralisedDiffusionCoefficientFit( ) )
			xend = self.motilitys[i].getGlobalEndTime()
			xstart = self.motilitys[i].getGlobalStartTime()
			if type_time == 'norm' :
				start_time.append( (xend-xstart)/(xend+xstart) )
			elif type_time == 'ageing':
				start_time.append( xstart/(xend-xstart) )
			elif type_time == 'init':
				start_time.append( xstart )
			elif type_time == 'middle':
				start_time.append( xstart + (xend-xstart)*0.5 )

		xsize = (6*len(self.motilitys),5) if is_multi else (6,5)
		x_label = ''
		if type_time == 'norm' :
			x_label = r'$\frac{t_f-t_i}{t_f+t_i}$'
		elif type_time == 'ageing':
			x_label = r'$t_i/T$'
		elif type_time == 'init':
			x_label = r'$t_i$'
		elif type_time == 'middle':
			x_label = r'$\frac{t_f-t_i}{2}$'

		return self._util.scheme_scatter( xdata = start_time, ydata = first_diffusion_fit_values, is_binned = is_binned, bins = bins, is_fit = is_fit, type_fit = type_fit, window = window, xlabel = x_label, ylabel = r'$K_1$', size = xsize, is_multi = is_multi, is_legend = is_legend, marker = marker, markersize = markersize, alpha = alpha, xscale = xscale , yscale = yscale )

	def showFirstScalingExponentTimeExperiment ( self, type_time = 'ageing', is_binned = False, bins = 20, is_set_fit_TAMSD = False, window_TAMSD = (0,6), is_fit = False, type_fit = 'linear', window = None, is_multi = False, is_legend = False, marker = 'o', markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear' ):
		if is_set_fit_TAMSD :
			for i in range( len(self.motilitys) ):
				self.motilitys[i].setFitTAMSD( window = window_TAMSD )

		start_time = []
		first_exponent_fit_values = []
		for i in range( len(self.motilitys) ):
			first_exponent_fit_values.append( self.motilitys[i].getFirstScalingExponentFit( ) )
			xend = self.motilitys[i].getGlobalEndTime()
			xstart = self.motilitys[i].getGlobalStartTime()
			if type_time == 'norm' :
				start_time.append( (xend-xstart)/(xend+xstart) )
			elif type_time == 'ageing':
				start_time.append( xstart/(xend-xstart) )
			elif type_time == 'init':
				start_time.append( xstart )
			elif type_time == 'middle':
				start_time.append( xstart + (xend-xstart)*0.5 )

		xsize = (6*len(self.motilitys),5) if is_multi else (6,5)
		x_label = ''
		if type_time == 'norm' :
			x_label = r'$\frac{t_f-t_i}{t_f+t_i}$'
		elif type_time == 'ageing':
			x_label = r'$t_i/T$'
		elif type_time == 'init':
			x_label = r'$t_i$'
		elif type_time == 'middle':
			x_label = r'$\frac{t_f-t_i}{2}$'

		return self._util.scheme_scatter( xdata = start_time, ydata = first_exponent_fit_values, is_binned = is_binned, bins = bins, is_fit = is_fit, type_fit = type_fit, window = window, xlabel = x_label, ylabel = r'${\beta}_1$', size = xsize, is_multi = is_multi, is_legend = is_legend, marker = marker, markersize = markersize, alpha = alpha, xscale = xscale , yscale = yscale )

	def showGeneralizedDiffusionCoefficientTimeExperiment ( self, type_time = 'ageing', is_binned = False, bins = 20, is_set_fit_TAMSD = False, window_TAMSD = (0,6), is_fit = False, type_fit = 'linear', window = None, is_multi = False, is_legend = False, marker = 'o', markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear' ):

		if is_set_fit_TAMSD :
			for i in range( len(self.motilitys) ):
				self.motilitys[i].setFitTAMSD( window = window_TAMSD )

		start_time = []
		diffusion_fit_values = []
		for i in range( len(self.motilitys) ):
			diffusion_fit_values.append( self.motilitys[i].getGeneralisedDiffusionCoefficientFit( ) )
			xend = self.motilitys[i].getGlobalEndTime()
			xstart = self.motilitys[i].getGlobalStartTime()
			if type_time == 'norm' :
				start_time.append( (xend-xstart)/(xend+xstart) )
			elif type_time == 'ageing':
				start_time.append( xstart/(xend-xstart) )
			elif type_time == 'init':
				start_time.append( xstart )
			elif type_time == 'middle':
				start_time.append( xstart + (xend-xstart)*0.5 )

		xsize = (6*len(self.motilitys),5) if is_multi else (6,5)
		x_label = ''
		if type_time == 'norm' :
			x_label = r'$\frac{t_f-t_i}{t_f+t_i}$'
		elif type_time == 'ageing':
			x_label = r'$t_i/T$'
		elif type_time == 'init':
			x_label = r'$t_i$'
		elif type_time == 'middle':
			x_label = r'$\frac{t_f-t_i}{2}$'

		return self._util.scheme_scatter( xdata = start_time, ydata = diffusion_fit_values, is_binned = is_binned, bins = bins, is_fit = is_fit, type_fit = type_fit, window = window, xlabel = x_label, ylabel = r'$K_{\beta}$', size = xsize, is_multi = is_multi, is_legend = is_legend, marker = marker, markersize = markersize, alpha = alpha, xscale = xscale , yscale = yscale )

	def showScalingExponentTimeExperiment ( self, type_time = 'ageing', is_binned = False, bins = 20, is_set_fit_TAMSD = False, window_TAMSD = (0,6), is_fit = False, type_fit = 'linear', window = None, is_multi = False, is_legend = False, marker = 'o', markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear' ):

		if is_set_fit_TAMSD :
			for i in range( len(self.motilitys) ):
				self.motilitys[i].setFitTAMSD( window = window_TAMSD )

		start_time = []
		exponent_fit_values = []
		for i in range( len(self.motilitys) ):
			exponent_fit_values.append( self.motilitys[i].getScalingExponentFit( ) )
			xend = self.motilitys[i].getGlobalEndTime()
			xstart = self.motilitys[i].getGlobalStartTime()
			if type_time == 'norm' :
				start_time.append( (xend-xstart)/(xend+xstart) )
			elif type_time == 'ageing':
				start_time.append( xstart/(xend-xstart) )
			elif type_time == 'init':
				start_time.append( xstart )
			elif type_time == 'middle':
				start_time.append( xstart + (xend-xstart)*0.5 )

		xsize = (6*len(self.motilitys),5) if is_multi else (6,5)
		x_label = ''
		if type_time == 'norm' :
			x_label = r'$\frac{t_f-t_i}{t_f+t_i}$'
		elif type_time == 'ageing':
			x_label = r'$t_i/T$'
		elif type_time == 'init':
			x_label = r'$t_i$'
		elif type_time == 'middle':
			x_label = r'$\frac{t_f-t_i}{2}$'
		return self._util.scheme_scatter( xdata = start_time, ydata = exponent_fit_values, is_binned = is_binned, bins = bins, is_fit = is_fit, type_fit = type_fit, window = window, xlabel = x_label, ylabel = r'$\beta$', size = xsize, is_multi = is_multi, is_legend = is_legend, marker = marker, markersize = markersize, alpha = alpha, xscale = xscale , yscale = yscale )

	# persistence with ageing

	def showFractalTimeExperiment ( self, type_time = 'ageing', is_binned = False, bins = 20, is_multi = False, is_fit = False, type_fit = 'linear', window = None, markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear', is_legend = False ) :

		start_time = []
		fractal = []
		for i in range( len(self.motilitys) ):
			fractal.append( self.motilitys[i].getGlobalFractalDimension() )
			xend = self.motilitys[i].getGlobalEndTime()
			xstart = self.motilitys[i].getGlobalStartTime()
			if type_time == 'norm' :
				start_time.append( (xend-xstart)/(xend+xstart) )
			elif type_time == 'ageing':
				start_time.append( xstart/(xend-xstart) )
			elif type_time == 'init':
				start_time.append( xstart )
			elif type_time == 'middle':
				start_time.append( xstart + (xend-xstart)*0.5 )

		xsize = (6*len(self.motilitys),5) if is_multi else (6,5)
		x_label = ''
		if type_time == 'norm' :
			x_label = r'$\frac{t_f-t_i}{t_f+t_i}$'
		elif type_time == 'ageing':
			x_label = r'$t_i/T$'
		elif type_time == 'init':
			x_label = r'$t_i$'
		elif type_time == 'middle':
			x_label = r'$\frac{t_f-t_i}{2}$'
		return self._util.scheme_scatter( xdata = start_time, ydata = fractal, is_binned = is_binned, bins = bins, is_fit = is_fit, type_fit = type_fit, window = window, xlabel = x_label , ylabel = r'$d_f$', is_multi = is_multi, size = xsize, markersize = markersize, alpha = alpha, xscale = xscale, yscale = yscale, is_legend = False )

	def showConfinementRatioTimeExperiment ( self, type_time = 'ageing', is_binned = False, bins = 20, is_multi = False, is_fit = False, type_fit = 'linear', window = None, markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear', is_legend = False ):

		start_time = []
		ratio_confinement = []
		for i in range( len(self.motilitys) ):
			ratio_confinement.append( self.motilitys[i].getGlobalPersistenceRatio() )
			xend = self.motilitys[i].getGlobalEndTime()
			xstart = self.motilitys[i].getGlobalStartTime()
			if type_time == 'norm' :
				start_time.append( (xend-xstart)/(xend+xstart) )
			elif type_time == 'ageing':
				start_time.append( xstart/(xend-xstart) )
			elif type_time == 'init':
				start_time.append( xstart )
			elif type_time == 'middle':
				start_time.append( xstart + (xend-xstart)*0.5 )

		xsize = (6*len(self.motilitys),5) if is_multi else (6,5)
		x_label = ''
		if type_time == 'norm' :
			x_label = r'$\frac{t_f-t_i}{t_f+t_i}$'
		elif type_time == 'ageing':
			x_label = r'$t_i/T$'
		elif type_time == 'init':
			x_label = r'$t_i$'
		elif type_time == 'middle':
			x_label = r'$\frac{t_f-t_i}{2}$'
		return self._util.scheme_scatter( xdata = start_time, ydata = ratio_confinement, is_binned = is_binned, bins = bins, is_fit = is_fit, type_fit = type_fit, window = window, xlabel = x_label , ylabel = 'Confinement ratio', is_multi = is_multi, size = xsize, markersize = markersize, alpha = alpha, xscale = xscale, yscale = yscale, is_legend = is_legend )

	# all TAMSD, MSD, Amplitude, EB with ageing

	def showTimeAverageMSDWithAgeing ( self, is_multi = False, is_fit = False, window = (0,20), xscale = 'linear', yscale = 'linear' ):

		def fit_tamsd(t, alpha, org):
			return alpha*t + org

		const_fits = []

		if is_multi :
			plt.figure( figsize = ( 8*len(self.motilitys), 5 ) )
		else:
			plt.figure( figsize = ( 8, 5 ) )

		time_tamsd = []
		y_tamsd = []
		time_etamsd = []
		avg_etamsd = []
		std_etamsd = []

		for i in range( len(self.motilitys) ):
			xtime_tamsd, xy_tamsd = self.motilitys[i].getAllTAMSDWithAgeing()
			xtime_etamsd, xavg_etamsd, xstd_etamsd = self.motilitys[i].getAverageTAMSDWithAgeing()

			time_tamsd.append( xtime_tamsd )
			y_tamsd.append( xy_tamsd )
			time_etamsd.append( xtime_etamsd )
			avg_etamsd.append( xavg_etamsd )
			std_etamsd.append( xstd_etamsd )
			
		for i in range( len(self.motilitys) ):

			if is_multi:
				plt.subplot(1, len(self.motilitys), i+1)
				plt.plot( time_tamsd[i], y_tamsd[i], color = self._colors[i] )
				plt.plot( time_etamsd[i], avg_etamsd[i], marker = 'o', markeredgecolor = 'black', markersize = 12, linestyle = '', markerfacecolor = self._colors[i], lw = 2, label = self._names[i] )
			else:
				plt.plot( time_tamsd[i], y_tamsd[i], color = self._colors[i] )

			if is_fit and is_multi :
				self.motilitys[i].setFitTAMSD( window = window )

				xf_etamsd = curve_fit( fit_tamsd, numpy.log10(time_etamsd[i][window[0]:window[1]]), numpy.log10( avg_etamsd[i][ window[0]:window[1] ] ) )
				
				plt.plot( time_etamsd[i][window[0]:window[1]],  numpy.power(time_etamsd[i][window[0]:window[1]], xf_etamsd[0][0] )*math.pow(10, xf_etamsd[0][1]), linestyle = ':', color = 'black', lw = 2 )
				plt.text( numpy.average(time_etamsd[i][0]), numpy.average(avg_etamsd[i][0]), ' slope '+str( round(xf_etamsd[0][0],2) ), color = self._colors[i] )

				const_fits.append( ( math.pow(10, xf_etamsd[0][1]), xf_etamsd[0][0] ) )
			if is_multi :
				plt.xscale( xscale )
				plt.yscale( yscale )
				plt.xticks( fontsize = self._font_size )
				plt.yticks( fontsize = self._font_size )
				plt.xlabel( 'time (min)', fontdict = { 'size' : self._font_size } )
				plt.ylabel( r'$TAMSD(\Delta;t_a,T), \; <TAMSD(\Delta;t_a,T)> (\mu m^2)$', fontdict = { 'size' : self._font_size } )
				plt.grid( linestyle = ':' )

		if not is_multi :
			for i in range( len(self.motilitys) ):
				plt.plot( time_etamsd[i], avg_etamsd[i], marker = 'o', markeredgecolor = 'black', markersize = 12, linestyle = '', markerfacecolor = self._colors[i], lw = 2, label = self._names[i] )
				
				if is_fit :
					self.motilitys[i].setFitTAMSD( window = window )

					xf_etamsd = curve_fit( fit_tamsd, numpy.log10(time_etamsd[i][window[0]:window[1]]), numpy.log10( avg_etamsd[i][ window[0]:window[1] ] ) )
					
					plt.plot( time_etamsd[i][window[0]:window[1]],  numpy.power(time_etamsd[i][window[0]:window[1]], xf_etamsd[0][0] )*math.pow(10, xf_etamsd[0][1]), linestyle = ':', color = 'black', lw = 2 )
					plt.text( numpy.average(time_etamsd[i][0]), numpy.average(avg_etamsd[i][0]), ' slope '+str( round(xf_etamsd[0][0],2) ), color = self._colors[i] )

					const_fits.append( ( math.pow(10, xf_etamsd[0][1]), xf_etamsd[0][0] ) )

			plt.xscale( xscale )
			plt.yscale( yscale )
			plt.xticks( fontsize = self._font_size )
			plt.yticks( fontsize = self._font_size )
			plt.xlabel( 'time (min)', fontdict = { 'size' : self._font_size } )
			plt.ylabel( r'TAMSD, <TAMSD> $(\mu m^2)$', fontdict = { 'size' : self._font_size } )
			plt.legend( frameon = False )
			plt.grid( linestyle = ':' )
				
		plt.tight_layout()

		return const_fits, plt

	def showErgodicityBreakingParameterTAMSDWithAgeing ( self, is_multi= False, is_color = False, is_legend = False, xscale = 'linear', yscale = 'linear' ) :

		time = []
		eb = []

		for i in range( len(self.motilitys) ):
			xtime, xeb, = self.motilitys[i].getErgodicityBreakingParameterTAMSDWithAgeing()
			i_nonnan = numpy.where( ~numpy.isnan(xeb) )[0]
			time.append( xtime/i_nonnan[-1] )
			eb.append( xeb )

		xsize = (6*len(self.motilitys), 5) if is_multi else (6, 5)
		xinfo_fit, xplt = self._util.scheme_scatter( xdata = time, ydata = eb, xlabel = r'$\Delta/T$', ylabel = 'EB', size = xsize, alpha = 0.5, is_multi = is_multi, is_color = is_color, is_legend = is_legend, xscale = xscale, yscale = yscale )
		if not is_multi:
			xplt.plot(time[0],(4/3)*time[0], color= 'black', linestyle = ':')

		return xinfo_fit, xplt

	def showHistAmplitudeTAMSDWithAgeing ( self, dtype = 'hist', ylim = None, is_multi = False, marker = 'o', markersize = 10, is_fit = False, type_fit = 'normal', htype = 'bar', xscale = 'linear', yscale = 'linear', ylabel = 'Count', bins = 10, is_density = False ):

		amplitude = []
		for i in range( len(self.motilitys) ):
			amplitude.append( self.motilitys[i].getAmplitudeTAMSDWithAging( is_flat_not_nan = True )	)
		
		if dtype == 'hist':
			return self._util.scheme_hist( data = amplitude, ylim = ylim, htype = htype, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$\xi$', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = amplitude, ylim = ylim, marker = marker, markersize = markersize, is_fit = is_fit, ylabel = ylabel, xlabel = r'$\xi$', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )

	def showMSDAndAverageTAMSDWithAgeing( self, is_fit = False, window = (0,20), xscale = 'linear', yscale = 'linear' ):

		def fit_tamsd(t, alpha, org):
			return alpha*t + org

		const_fits = []

		plt.figure( figsize = ( 8, 5 ) )

		for i in range( len(self.motilitys) ):
			time_etamsd, avg_etamsd, std_etamsd = self.motilitys[i].getAverageTAMSDWithAgeing()
			time_emsd, avg_emsd, std_emsd = self.motilitys[i].getMSDWithAgeing()

			plt.plot( time_etamsd, avg_etamsd, marker = 'o', markeredgecolor = self._colors[i], markersize = 12, linestyle = '', markerfacecolor = (1,1,1,0), lw = 2 )
			plt.plot( time_emsd, avg_emsd, marker = 's', markeredgecolor = self._colors[i], markersize = 12, linestyle = '', markerfacecolor = (1,1,1,0), lw = 2 )
			
			if is_fit:

				x_fit_etamsd = time_etamsd[window[0]:window[1]]
				y_fit_etamsd = avg_etamsd[window[0]:window[1]]
				i_nonnan = numpy.where( ~numpy.isnan(y_fit_etamsd) )[0]
				x_fit_etamsd = x_fit_etamsd[i_nonnan]
				y_fit_etamsd = y_fit_etamsd[i_nonnan]

				x_fit_emsd = time_emsd[window[0]:window[1]]
				y_fit_emsd = avg_emsd[window[0]:window[1]]
				i_nonnan = numpy.where( ~numpy.isnan(y_fit_emsd) )[0]
				x_fit_emsd = x_fit_emsd[i_nonnan]
				y_fit_emsd = y_fit_emsd[i_nonnan]

				xf_etamsd = curve_fit( fit_tamsd, numpy.log10(x_fit_etamsd), numpy.log10(y_fit_etamsd) )
				xf_emsd = curve_fit( fit_tamsd, numpy.log10(x_fit_emsd), numpy.log10(y_fit_emsd) )

				plt.plot( x_fit_etamsd,  numpy.power(x_fit_etamsd, xf_etamsd[0][0] )*math.pow(10, xf_etamsd[0][1]), linestyle = '-', color = self._colors[i], lw = 2 )
				plt.text( time_etamsd[0], avg_etamsd[0], r'$slope_{<TAMSD>}$ ~ '+str( round(xf_etamsd[0][0],2) ), color = self._colors[i] )

				plt.plot( x_fit_emsd,  numpy.power(x_fit_emsd, xf_emsd[0][0] )*math.pow(10, xf_emsd[0][1]), linestyle = ':', color = self._colors[i], lw = 2 )
				plt.text( time_emsd[1], avg_emsd[1], r'$slope_{MSD}$ ~ '+str( round(xf_emsd[0][0],2) ), color = self._colors[i] )
				
				const_fits.append( ( math.pow(10, xf_etamsd[0][1]), xf_etamsd[0][0], math.pow(10, xf_emsd[0][1]), xf_emsd[0][0] ) )

		plt.xscale( xscale )
		plt.yscale( yscale )
		plt.xticks( fontsize = self._font_size )
		plt.yticks( fontsize = self._font_size )
		plt.xlabel( 'time (min)', fontdict = { 'size' : self._font_size } )
		plt.ylabel( r'MSD, <TAMSD> $(\mu m^2)$', fontdict = { 'size' : self._font_size } )
		plt.grid( linestyle = ':' )

		return const_fits, plt

	def showTAMSDMSDRatioWithAgeing ( self, is_multi= False, is_color = False, is_legend = False, xscale = 'linear', yscale = 'linear' ) :

		time = []
		eb = []
				
		for i in range( len(self.motilitys) ):
			xtime, xeb, = self.motilitys[i].getTAMSDMSDRatioWithAgeing()	
			time.append( xtime )
			eb.append( xeb )

		xsize = (6*len(self.motilitys), 5) if is_multi else (6, 5)
		xinfo_fit, xplt = self._util.scheme_scatter( xdata = time, ydata = eb, xlabel = r'$\Delta \; (min)$', ylabel = r'$\mathcal{EB}$', size = xsize, alpha = 0.5, is_multi = is_multi, is_color = is_color, is_legend = is_legend, xscale = xscale, yscale = yscale )
		if not is_multi:
			i_nonnan = numpy.where( ~numpy.isnan(eb[0]) )[0]
			xplt.plot( [0, time[0][i_nonnan[-1]] ], [1,1], color = 'black', linestyle = ':' )

		return xinfo_fit, xplt

	# dynamical functional

	def showXAllMixingErgoDynamicalFunctionalTest ( self, ylabel_mix_r = r'$Re\left( \hat E(n) \right)$', ylabel_mix_i = r'$Img\left( \hatE(n) \right)$', ylabel_erg_r = r'$Re\left( \sum_{k=0}^{n-1} \hat E(k)/n \right)$', ylabel_erg_i = r'$Img\left( \sum_{k=0}^{n-1} \hat E(k)/n \right)$' ):

		plt.figure( figsize = ( 24, 4*len(self.motilitys) ) )

		for i in range( len(self.motilitys) ):
			xtim, xmix, xergo = self.motilitys[i].getXAllMixingBreakingDynamicalFunctionalTest()
			plt.subplot(len(self.motilitys),4,4*i+1)
			self._util.scheme_plot_fill( xdata = [xtim], ydata = [numpy.real(xmix)], ystd = [], is_fig = False, xlabel = 'Time (min)', ylabel = ylabel_mix_r, is_color = True )
			plt.subplot(len(self.motilitys),4,4*i+2)
			self._util.scheme_plot_fill( xdata = [xtim], ydata = [numpy.imag(xmix)], ystd = [], is_fig = False, xlabel = 'Time (min)', ylabel = ylabel_mix_i, is_color = True )
			plt.subplot(len(self.motilitys),4,4*i+3)
			self._util.scheme_plot_fill( xdata = [xtim], ydata = [numpy.real(xergo)], ystd = [], is_fig = False, xlabel = 'Time (min)', ylabel = ylabel_erg_r, is_color = True )
			plt.subplot(len(self.motilitys),4,4*i+4)
			self._util.scheme_plot_fill( xdata = [xtim], ydata = [numpy.imag(xergo)], ystd = [], is_fig = False, xlabel = 'Time (min)', ylabel = ylabel_erg_i, is_color = True )

		plt.tight_layout()
		return plt

	def showXAverageMixingErgodicityDynamicalFunctionalTest ( self, ylabel_mix_r = 'Re(E(n))', ylabel_mix_i = 'Img(E(n))', ylabel_erg_r = r'$Re\left( \sum_{k=0}^{n-1} E(k)/n \right)$', ylabel_erg_i = r'$Img\left( \sum_{k=0}^{n-1} E(k)/n \right)$', is_legend = False ):

		time = []
		mixing_real = []
		mixing_imag = []
		ergodicity_real = []
		ergodicity_imag = []
				
		for i in range( len(self.motilitys) ):
			xtime, xmixing, xergo = self.motilitys[i].getXAverageMixingBreakingDynamicalFunctionalTest()

			time.append( xtime )
			mixing_real.append( xmixing.real )
			mixing_imag.append( xmixing.imag )
			ergodicity_real.append( xergo.real )
			ergodicity_imag.append( xergo.imag )


		plt.figure( figsize = ( 12, 8 ) )
		plt.subplot(221)
		self._util.scheme_plot_fill( xdata = time, ydata = mixing_real, is_fig = False, xlabel = 't (min)', ylabel = ylabel_mix_r, is_legend = is_legend )
		plt.subplot(222)
		self._util.scheme_plot_fill( xdata = time, ydata = mixing_imag, is_fig = False, xlabel = 't (min)', ylabel = ylabel_mix_i, is_legend = is_legend )
		plt.subplot(223)
		self._util.scheme_plot_fill( xdata = time, ydata = ergodicity_real, is_fig = False, xlabel = 't (min)', ylabel = ylabel_erg_r, is_legend = is_legend )
		plt.subplot(224)
		self._util.scheme_plot_fill( xdata = time, ydata = ergodicity_imag, is_fig = False, xlabel = 't (min)', ylabel = ylabel_erg_i, is_legend = is_legend )

		plt.tight_layout()

		return plt				

	def showXErgodicityDynamicalFunctionalTestScalingExponent ( self, window_TAMSD = (0,10), is_multi = False, is_fit = False, is_fit_all = False, type_fit = 'linear', window = None, alpha = 0.5, marker = 'o', markersize = 100, xscale = 'linear', yscale = 'linear' ):

		exponent = []
		mixing = []
		ergodicity = []
		for i in range( len(self.motilitys) ):
			xtim, xmixing, xergo = self.motilitys[i].getXAllMixingBreakingDynamicalFunctionalTest()
			xend_index = self.motilitys[i].getGlobalEndIndex()
			xstart_index = self.motilitys[i].getGlobalStartIndex()
			xscaling, xcoefficient = self.motilitys[i].getXAllScalingExponentAndGeneraisedDiffusionCoefficient( window = window_TAMSD )

			exponent.append( xscaling )
			mixing.append( numpy.real(xmixing)[ xend_index-xstart_index-1, range( xmixing.shape[1] ) ] )
			ergodicity.append( numpy.real(xergo)[ xend_index-xstart_index-1, range( xergo.shape[1] ) ] )

		xsize = (6*len(self.motilitys), 5) if is_multi else (6, 5)
		xinfo_mix, xpltmix = self._util.scheme_scatter( xdata = exponent, ydata = mixing, is_multi = is_multi, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit , window = window, alpha = alpha, marker = marker, markersize = markersize, size = xsize, xlabel = r'$\beta$', ylabel = r'$Re\left( \hat E(n) \right)$', xscale = xscale, yscale = yscale )
		xinfo_ergo, xpltergo = self._util.scheme_scatter( xdata = exponent, ydata = ergodicity, is_multi = is_multi, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit , window = window, alpha = alpha, marker = marker, markersize = markersize, size = xsize, xlabel = r'$\beta$', ylabel = r'$Re\left( \sum_{k=0}^{n-1} \hat E(k)/n \right)$', xscale = xscale, yscale = yscale )

		return xinfo_mix, xpltmix, xinfo_ergo, xpltergo

	def showYAllMixingErgoDynamicalFunctionalTest ( self, ylabel_mix_r = r'$Re\left( \hat E(n) \right)$', ylabel_mix_i = r'$Img\left( \hatE(n) \right)$', ylabel_erg_r = r'$Re\left( \sum_{k=0}^{n-1} \hat E(k)/n \right)$', ylabel_erg_i = r'$Img\left( \sum_{k=0}^{n-1} \hat E(k)/n \right)$' ):

		plt.figure( figsize = ( 24, 4*len(self.motilitys) ) )

		for i in range( len(self.motilitys) ):
			ytim, ymix, yergo = self.motilitys[i].getYAllMixingBreakingDynamicalFunctionalTest()
			plt.subplot(len(self.motilitys),4,4*i+1)
			self._util.scheme_plot_fill( xdata = [ytim], ydata = [numpy.real(ymix)], ystd = [], is_fig = False, xlabel = 'Time (min)', ylabel = ylabel_mix_r, is_color = True )
			plt.subplot(len(self.motilitys),4,4*i+2)
			self._util.scheme_plot_fill( xdata = [ytim], ydata = [numpy.imag(ymix)], ystd = [], is_fig = False, xlabel = 'Time (min)', ylabel = ylabel_mix_i, is_color = True )
			plt.subplot(len(self.motilitys),4,4*i+3)
			self._util.scheme_plot_fill( xdata = [ytim], ydata = [numpy.real(yergo)], ystd = [], is_fig = False, xlabel = 'Time (min)', ylabel = ylabel_erg_r, is_color = True )
			plt.subplot(len(self.motilitys),4,4*i+4)
			self._util.scheme_plot_fill( xdata = [ytim], ydata = [numpy.imag(yergo)], ystd = [], is_fig = False, xlabel = 'Time (min)', ylabel = ylabel_erg_i, is_color = True )

		plt.tight_layout()
		return plt
	
	def showYAverageMixingErgodicityDynamicalFunctionalTest ( self, ylabel_mix_r = 'Re(E(n))', ylabel_mix_i = 'Img(E(n))', ylabel_erg_r = r'$Re\left( \sum_{k=0}^{n-1} E(k)/n \right)$', ylabel_erg_i = r'$Img\left( \sum_{k=0}^{n-1} E(k)/n \right)$', is_legend = False ):

		time = []
		mixing_real = []
		mixing_imag = []
		ergodicity_real = []
		ergodicity_imag = []
				
		for i in range( len(self.motilitys) ):
			ytime, ymixing, yergo = self.motilitys[i].getYAverageMixingBreakingDynamicalFunctionalTest()	
			time.append( ytime )
			mixing_real.append( ymixing.real )
			mixing_imag.append( ymixing.imag )
			ergodicity_real.append( yergo.real )
			ergodicity_imag.append( yergo.imag )


		plt.figure( figsize = ( 12, 8 ) )
		plt.subplot(221)
		self._util.scheme_plot_fill( xdata = time, ydata = mixing_real, is_fig = False, xlabel = 't (min)', ylabel = ylabel_mix_r, is_legend = is_legend )
		plt.subplot(222)
		self._util.scheme_plot_fill( xdata = time, ydata = mixing_imag, is_fig = False, xlabel = 't (min)', ylabel = ylabel_mix_i, is_legend = is_legend )
		plt.subplot(223)
		self._util.scheme_plot_fill( xdata = time, ydata = ergodicity_real, is_fig = False, xlabel = 't (min)', ylabel = ylabel_erg_r, is_legend = is_legend )
		plt.subplot(224)
		self._util.scheme_plot_fill( xdata = time, ydata = ergodicity_imag, is_fig = False, xlabel = 't (min)', ylabel = ylabel_erg_i, is_legend = is_legend )

		plt.tight_layout()

		return plt				

	def showYErgodicityDynamicalFunctionalTestScalingExponent ( self, window_TAMSD = (0,10), is_multi = False, is_fit = False, is_fit_all = False, type_fit = 'linear', window = None, alpha = 0.5, marker = 'o', markersize = 100, xscale = 'linear', yscale = 'linear' ):

		exponent = []
		mixing = []
		ergodicity = []
		for i in range( len(self.motilitys) ):
			ytim, ymixing, yergo = self.motilitys[i].getYAllMixingBreakingDynamicalFunctionalTest()
			yend_index = self.motilitys[i].getGlobalEndIndex()
			ystart_index = self.motilitys[i].getGlobalStartIndex()
			yscaling, ycoefficient = self.motilitys[i].getYAllScalingExponentAndGeneraisedDiffusionCoefficient( window = window_TAMSD )

			exponent.append( yscaling )
			mixing.append( numpy.real(ymixing)[ yend_index-ystart_index-1, range( ymixing.shape[1] ) ] )
			ergodicity.append( numpy.real(yergo)[ yend_index-ystart_index-1, range( yergo.shape[1] ) ] )

		xsize = (6*len(self.motilitys), 5) if is_multi else (6, 5)
		yinfo_mix, ypltmix = self._util.scheme_scatter( xdata = exponent, ydata = mixing, is_multi = is_multi, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit , window = window, alpha = alpha, marker = marker, markersize = markersize, size = xsize, xlabel = r'$\beta$', ylabel = r'$Re\left( \hat E(n) \right)$', xscale = xscale, yscale = yscale )
		yinfo_ergo, ypltergo = self._util.scheme_scatter( xdata = exponent, ydata = ergodicity, is_multi = is_multi, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit , window = window, alpha = alpha, marker = marker, markersize = markersize, size = xsize, xlabel = r'$\beta$', ylabel = r'$Re\left( \sum_{k=0}^{n-1} \hat E(k)/n \right)$', xscale = xscale, yscale = yscale )

		return yinfo_mix, ypltmix, yinfo_ergo, ypltergo

	# all persistence

	def showDiffusionCoefficientConfinementRatio ( self, is_set_fit_TAMSD = False, window_TAMSD = (1,6), is_fit = False,is_fit_all = False, type_fit = 'linear', window = None, is_multi = False, markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear', is_legend = False ):
		
		if is_set_fit_TAMSD :
			for i in range( len(self.motilitys) ):
				self.motilitys[i].setFitTAMSD( window = window_TAMSD )
		
		ratio_confinement = []
		diffussion_fit_values = []
		for i in range( len(self.motilitys) ):
			ratio_confinement.append( self.motilitys[i].getGlobalPersistenceRatio() )
			diffussion_fit_values.append( self.motilitys[i].getGeneralisedDiffusionCoefficientFit() )

		xsize = ( 6*len(ratio_confinement) ,5) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = ratio_confinement, ydata = diffussion_fit_values, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit, window = window, xlabel = 'Confinement ratio' , ylabel = r'$ K_{\beta} $', is_multi = is_multi, size = xsize, markersize = markersize, alpha = alpha, xscale = xscale, yscale = yscale, is_legend = is_legend )

	def showScalingExponentConfinementRatio ( self, is_set_fit_TAMSD = False, window_TAMSD = (1,6), is_fit = False, is_fit_all = False, type_fit = 'linear', window = None, is_multi = False, markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear', is_legend = False ):
		
		if is_set_fit_TAMSD :
			for i in range( len(self.motilitys) ):
				self.motilitys[i].setFitTAMSD( window = window_TAMSD )
		
		ratio_confinement = []
		exponent_fit_values = []
		for i in range( len(self.motilitys) ):
			ratio_confinement.append( self.motilitys[i].getGlobalPersistenceRatio() )
			exponent_fit_values.append( self.motilitys[i].getScalingExponentFit() )

		xsize = ( 6*len(ratio_confinement) ,5) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = ratio_confinement, ydata = exponent_fit_values, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit, window = window, xlabel = 'Confinement ratio' , ylabel = r'$\beta$', is_multi = is_multi, size = xsize, markersize = markersize, alpha = alpha, xscale = xscale, yscale = yscale, is_legend = is_legend )
	
	def showDiffusionCoefficientFractalDimension ( self, is_set_fit_TAMSD = False, window_TAMSD = (1,6), is_fit = False, is_fit_all = False, type_fit = 'linear', window = None, is_multi = False, markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear', is_legend = False ):

		if is_set_fit_TAMSD :
			for i in range( len(self.motilitys) ):
				self.motilitys[i].setFitTAMSD( window = window_TAMSD )
		
		fractal = []
		diffussion_fit_values = []
		for i in range( len(self.motilitys) ):
			fractal.append( self.motilitys[i].getGlobalFractalDimension() )
			diffussion_fit_values.append( self.motilitys[i].getGeneralisedDiffusionCoefficientFit() )

		xsize = ( 6*len(diffussion_fit_values) ,5) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = fractal, ydata = diffussion_fit_values, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit, window = window, xlabel = r'$d_f$' , ylabel = r'$ K_{\beta} $', is_multi = is_multi, size = xsize, markersize = markersize, alpha = alpha, xscale = xscale, yscale = yscale, is_legend = is_legend )

	def showScalingExponentFractalDimension ( self, is_set_fit_TAMSD = False, window_TAMSD = (1,6), is_fit = False, is_fit_all = False, type_fit = 'linear', window = None, is_multi = False, markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear', is_legend = False ) :

		if is_set_fit_TAMSD :
			for i in range( len(self.motilitys) ):
				self.motilitys[i].setFitTAMSD( window = window_TAMSD )
		
		exponent_fit_values = []
		fractal = []
		for i in range( len(self.motilitys) ):
			exponent_fit_values.append( self.motilitys[i].getScalingExponentFit() )
			fractal.append( self.motilitys[i].getGlobalFractalDimension() )

		xsize = ( 6*len(exponent_fit_values) ,5) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = fractal, ydata = exponent_fit_values, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit, window = window, xlabel = r'$d_f$' , ylabel = r'$\beta$', is_multi = is_multi, size = xsize, markersize = markersize, alpha = alpha, xscale = xscale, yscale = yscale, is_legend = is_legend )

	def showDiffusionCoefficientDisplacementRatio ( self, is_set_fit_TAMSD = False, window_TAMSD = (1,6), is_fit = False,is_fit_all = False, type_fit = 'linear', window = None, is_multi = False, markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear', is_legend = False ):
		
		if is_set_fit_TAMSD :
			for i in range( len(self.motilitys) ):
				self.motilitys[i].setFitTAMSD( window = window_TAMSD )
		
		ratio_displacement = []
		diffussion_fit_values = []
		for i in range( len(self.motilitys) ):
			ratio_displacement.append( self.motilitys[i].getGlobalDisplacementRatio() )
			diffussion_fit_values.append( self.motilitys[i].getGeneralisedDiffusionCoefficientFit() )

		xsize = ( 6*len(ratio_displacement) ,5) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = ratio_displacement, ydata = diffussion_fit_values, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit, window = window, xlabel = 'Displacement ratio' , ylabel = r'$ K_{\beta} $', is_multi = is_multi, size = xsize, markersize = markersize, alpha = alpha, xscale = xscale, yscale = yscale, is_legend = is_legend )

	def showScalingExponentDisplacementRatio ( self, is_set_fit_TAMSD = False, window_TAMSD = (1,6), is_fit = False, is_fit_all = False, type_fit = 'linear', window = None, is_multi = False, markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear', is_legend = False ):
		
		if is_set_fit_TAMSD :
			for i in range( len(self.motilitys) ):
				self.motilitys[i].setFitTAMSD( window = window_TAMSD )
		
		ratio_displacement = []
		exponent_fit_values = []
		for i in range( len(self.motilitys) ):
			ratio_displacement.append( self.motilitys[i].getGlobalDisplacementRatio() )
			exponent_fit_values.append( self.motilitys[i].getScalingExponentFit() )

		xsize = ( 6*len(ratio_displacement) ,5) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = ratio_displacement, ydata = exponent_fit_values, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit, window = window, xlabel = 'Displacement ratio' , ylabel = r'$\beta$', is_multi = is_multi, size = xsize, markersize = markersize, alpha = alpha, xscale = xscale, yscale = yscale, is_legend = is_legend )

	def showDiffusionCoefficientOutreachRatio ( self, is_set_fit_TAMSD = False, window_TAMSD = (1,6), is_fit = False,is_fit_all = False, type_fit = 'linear', window = None, is_multi = False, markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear', is_legend = False ):
		
		if is_set_fit_TAMSD :
			for i in range( len(self.motilitys) ):
				self.motilitys[i].setFitTAMSD( window = window_TAMSD )
		
		ratio_outreach = []
		diffussion_fit_values = []
		for i in range( len(self.motilitys) ):
			ratio_outreach.append( self.motilitys[i].getGlobalOutreachRatio() )
			diffussion_fit_values.append( self.motilitys[i].getGeneralisedDiffusionCoefficientFit() )

		xsize = ( 6*len(ratio_outreach) ,5) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = ratio_outreach, ydata = diffussion_fit_values, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit, window = window, xlabel = 'Outreach ratio' , ylabel = r'$ K_{\beta} $', is_multi = is_multi, size = xsize, markersize = markersize, alpha = alpha, xscale = xscale, yscale = yscale, is_legend = is_legend )

	def showScalingExponentOutreachRatio ( self, is_set_fit_TAMSD = False, window_TAMSD = (1,6), is_fit = False, is_fit_all = False, type_fit = 'linear', window = None, is_multi = False, markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear', is_legend = False ):
		
		if is_set_fit_TAMSD :
			for i in range( len(self.motilitys) ):
				self.motilitys[i].setFitTAMSD( window = window_TAMSD )
		
		ratio_outreach = []
		exponent_fit_values = []
		for i in range( len(self.motilitys) ):
			ratio_outreach.append( self.motilitys[i].getGlobalOutreachRatio() )
			exponent_fit_values.append( self.motilitys[i].getScalingExponentFit() )

		xsize = ( 6*len(ratio_outreach) ,5) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = ratio_outreach, ydata = exponent_fit_values, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit, window = window, xlabel = 'Outreach ratio' , ylabel = r'$\beta$', is_multi = is_multi, size = xsize, markersize = markersize, alpha = alpha, xscale = xscale, yscale = yscale, is_legend = is_legend )

	def showFractalDimensionConfinementRatio ( self, is_multi = False, is_fit = False, is_fit_all = False, type_fit = 'linear', window = None, markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear', is_legend = False ) :

		ratio_confinement = []
		fractal = []
		for i in range( len(self.motilitys) ):
			ratio_confinement.append( self.motilitys[i].getGlobalPersistenceRatio() )
			fractal.append( self.motilitys[i].getGlobalFractalDimension() )

		xsize = ( 6*len(ratio_confinement) ,5) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = ratio_confinement, ydata = fractal, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit, window = window, xlabel = 'Confinement ratio' , ylabel = r'$d_f$', is_multi = is_multi, size = xsize, markersize = markersize, alpha = alpha, xscale = xscale, yscale = yscale, is_legend = is_legend )

	def showHistFractalDimension ( self, dtype = 'hist', ylim = None, is_multi = False, marker = 'o', markersize = 10, is_fit = False, type_fit = 'normal', htype = 'bar', xscale = 'linear', yscale = 'linear', bins = 10, is_density = False, is_legend = False ):
		
		fractal = []
		for i in range( len(self.motilitys) ):
			fractal.append( self.motilitys[i].getGlobalFractalDimension( ) )
		
		y_label = r'P( $d_f$ )' if is_density else r'N( $d_f$ )'
		if dtype == 'hist' :
			return self._util.scheme_hist( data = fractal, ylim = ylim, htype = htype, is_fit = is_fit, type_fit = type_fit, ylabel = y_label, xlabel = r'$d_f$', density = is_density, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = fractal, ylim = ylim, marker = marker, markersize = markersize, is_fit = is_fit, type_fit = type_fit, ylabel = y_label, xlabel = r'$d_f$', density = is_density, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )

	def showInterQuartileFractalDimension ( self, is_test = False, showfliers = True ):

		fractal = []
		for i in range( len(self.motilitys) ):
			fractal.append( self.motilitys[i].getGlobalFractalDimension( ) )

		return self._util.scheme_single_boxplot( data = fractal, ylabel = r'$d_f$', is_test = is_test, showfliers = showfliers )

	def showHistPersistenceRatio ( self, dtype = 'hist', ylim = None, is_multi = False, marker = 'o', markersize = 10, is_fit = False, type_fit = 'normal', htype = 'bar', xscale = 'linear', yscale = 'linear', bins = 10, is_density = False, is_legend = False ):

		persistenceratio = []
		for i in range( len(self.motilitys) ):
			persistenceratio.append( self.motilitys[i].getGlobalPersistenceRatio( is_not_nan = True ) )

		y_label = 'P( Confinement ratio )' if is_density else 'N( Confinement ratio )'
		if dtype == 'hist' :
			return self._util.scheme_hist( data = persistenceratio, ylim = ylim, htype = htype, is_fit = is_fit, type_fit = type_fit, ylabel = y_label, xlabel = 'Confinement ratio', density = is_density, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = persistenceratio, ylim = ylim, marker = marker, markersize = markersize, is_fit = is_fit, type_fit = type_fit, ylabel = y_label, xlabel = 'Confinement ratio', density = is_density, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )

	def showInterQuartilePersistenceRatio ( self, is_test = False, is_ns_test = False, showfliers = True ):

		persistenceratio = []
		for i in range( len(self.motilitys) ):
			persistenceratio.append( self.motilitys[i].getGlobalPersistenceRatio( is_not_nan = True ) )

		return self._util.scheme_single_boxplot( data = persistenceratio, ylabel = 'Confinement ratio', is_test = is_test, is_ns_test = is_ns_test, showfliers = showfliers )

	def showHistGyrationRadius ( self, dtype = 'hist', ylim = None, is_multi = False, marker = 'o', markersize = 10, is_fit = False, type_fit = 'normal', htype = 'bar', xscale = 'linear', yscale = 'linear', bins = 10, is_density = False, is_legend = False ):

		gyration_radius = []
		for i in range( len(self.motilitys) ):
			gyration_radius.append( self.motilitys[i].getGyrationRadius() )
		
		y_label = r'P( $R_G^2$ )' if is_density else r'N( $R_G^2$ )'
		if dtype == 'hist':
			return self._util.scheme_hist( data = gyration_radius, ylim = ylim, htype = htype, is_fit = is_fit, type_fit = type_fit, ylabel = y_label, xlabel = r'$R_G^2 \; (\mu m ^ 2)$', density = is_density, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = gyration_radius, ylim = ylim, marker = marker, markersize = markersize, is_fit = is_fit, type_fit = type_fit, ylabel = y_label, xlabel = r'$R_G^2 \; (\mu m ^ 2)$', density = is_density, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )

	def showHistGyrationAsymmetryRatio_a2 ( self, dtype = 'hist', ylim = None, is_multi = False, marker = 'o', markersize = 10, is_fit = False, type_fit = 'normal', htype = 'bar', xscale = 'linear', yscale = 'linear', ylabel = 'Count', bins = 10, is_density = False, is_legend = False ):

		asymmetryratio_a2 = []
		for i in range( len(self.motilitys) ):
			asymmetryratio_a2.append( self.motilitys[i].getGyrationAsymmetryRatio_a2( ) )
		
		if dtype == 'hist' :
			return self._util.scheme_hist( data = asymmetryratio_a2, ylim = ylim, htype = htype, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$a_2$', density = is_density, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = asymmetryratio_a2, ylim = ylim, marker = marker, markersize = markersize, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$a_2$', density = is_density, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )		

	def showHistGyrationAsymmetryRatio_A2 ( self, dtype = 'hist', ylim = None, is_multi = False, marker = 'o', markersize = 10, is_fit = False, type_fit = 'normal', htype = 'bar', xscale = 'linear', yscale = 'linear', ylabel = 'Count', bins = 10, is_density = False, is_legend = False ):

		asymmetryratio_A2 = []
		for i in range( len(self.motilitys) ):
			asymmetryratio_A2.append( self.motilitys[i].getGyrationAsymmetryRatio_A2( ) )
		
		if dtype == 'hist' :
			return self._util.scheme_hist( data = asymmetryratio_A2, ylim = ylim, htype = htype, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$A_2$', density = is_density, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = asymmetryratio_A2, ylim = ylim, marker = marker, markersize = markersize, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$A_2$', density = is_density, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )

	def showHistGyrationAsymmetry_A ( self, dtype = 'hist', ylim = None, is_multi = False, marker = 'o', markersize = 10, is_fit = False, type_fit = 'normal', htype = 'bar', xscale = 'linear', yscale = 'linear', ylabel = 'Count', bins = 10, is_density = False, is_legend = False ):

		asymmetry_A = []
		for i in range( len(self.motilitys) ):
			asymmetry_A.append( self.motilitys[i].getGyrationAsummetry_A( ) )
		
		if dtype == 'hist' :
			return self._util.scheme_hist( data = asymmetry_A, ylim = ylim, htype = htype, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = 'A', density = is_density, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = asymmetry_A, ylim = ylim, marker = marker, markersize = markersize, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = 'A', density = is_density, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )

	def showHistConvexHullPerimeter ( self, dtype = 'hist', ylim = None, is_multi = False, marker = 'o', markersize = 10, is_fit = False, type_fit = 'normal', htype = 'bar', xscale = 'linear', yscale = 'linear', bins = 10, is_density = False, is_legend = False ):

		convex_hull_perimeter = []
		for i in range( len(self.motilitys) ):
			convex_hull_perimeter.append( self.motilitys[i].getGlobalHullPerimeter( ) )
		
		y_label = 'P( A )' if is_density else 'N( A )'
		if dtype == 'hist' :
			return self._util.scheme_hist( data = convex_hull_perimeter, ylim = ylim, htype = htype, is_fit = is_fit, type_fit = type_fit, ylabel = y_label, xlabel = r'$P \; (\mu m)$', density = is_density, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = convex_hull_perimeter, ylim = ylim, marker = marker, markersize = markersize, is_fit = is_fit, type_fit = type_fit, ylabel = y_label, xlabel = r'$P \; (\mu m)$', density = is_density, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )

	def showInterQuartileConvexHullPerimeter ( self, is_test = False, showfliers = True ):

		convex_hull_perimeter = []
		for i in range( len(self.motilitys) ):
			convex_hull_perimeter.append( self.motilitys[i].getGlobalHullPerimeter( is_not_nan = True ) )

		return self._util.scheme_single_boxplot( data = convex_hull_perimeter, ylabel = r'P ($\mu m$)', is_test = is_test, showfliers = showfliers )

	def showHistConvexHullArea ( self, dtype = 'hist', ylim = None, is_multi = False, marker = 'o', markersize = 10, is_fit = False, type_fit = 'normal', htype = 'bar', xscale = 'linear', yscale = 'linear', bins = 10, is_density = False, is_legend = False ):

		convex_hull_area = []
		for i in range( len(self.motilitys) ):
			convex_hull_area.append( self.motilitys[i].getGlobalHullArea( ) )
		
		y_label = 'P( P )' if is_density else 'N( P )'		
		if dtype == 'hist' :
			return self._util.scheme_hist( data = convex_hull_area, ylim = ylim, htype = htype, is_fit = is_fit, type_fit = type_fit, ylabel = y_label, xlabel = r'$A \; (\mu m^2)$', density = is_density, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = convex_hull_area, ylim = ylim, marker = marker, markersize = markersize, is_fit = is_fit, type_fit = type_fit, ylabel = y_label, xlabel = r'$A \; (\mu m^2)$', density = is_density, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
	
	def showInterQuartileConvexHullArea ( self, is_test = False, showfliers = True ):

		convex_hull_area = []
		for i in range( len(self.motilitys) ):
			convex_hull_area.append( self.motilitys[i].getGlobalHullArea( is_not_nan = True ) )

		return self._util.scheme_single_boxplot( data = convex_hull_area, ylabel = r'A ($\mu m^2$)', is_test = is_test, showfliers = showfliers )

	def showHistConvexHullAcircularity ( self, dtype = 'hist', ylim = None, is_multi = False, marker = 'o', markersize = 10, is_fit = False, type_fit = 'normal', htype = 'bar', xscale = 'linear', yscale = 'linear', ylabel = 'Count', bins = 10, is_density = False, is_legend = False ):

		convex_hull_acir = []
		for i in range( len(self.motilitys) ):
			convex_hull_acir.append( self.motilitys[i].getGlobalAcircularity( ) )
		
		if dtype == 'hist' :
			return self._util.scheme_hist( data = convex_hull_acir, ylim = ylim, htype = htype, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = 'a', density = is_density, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = convex_hull_acir, ylim = ylim, marker = marker, markersize = markersize, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = 'a', density = is_density, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )

	def showAverageDistancePersistence ( self, show_std = True ):
		timepersistence = []
		avgpersistence = []
		stdpersistence = []
		for i in range( len(self.motilitys) ):
			time, average, std = self.motilitys[i].getAveragePersistenceDistance()		
			timepersistence.append( time )
			avgpersistence.append( average )
			if show_std :
				stdpersistence.append( std )

		return self._util.scheme_plot_fill( xdata = timepersistence, ydata = avgpersistence, ystd = stdpersistence, xlabel = 'Time (min)', ylabel = 'Temporal confinement ratio', is_legend = True )
	
	# Generalized Diffusion Coefficient 

	def showDiffusionCoefficientConvexHullArea ( self, is_set_fit_TAMSD = False, window_TAMSD = (1,6), is_fit = False, is_fit_all = False, type_fit = 'linear', window = None, is_multi = False, markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear', is_legend = False ):

		if is_set_fit_TAMSD :
			for i in range( len(self.motilitys) ):
				self.motilitys[i].setFitTAMSD( window = window_TAMSD )
		
		convexhullarea = []
		diffussion_fit_values = []
		for i in range( len(self.motilitys) ):
			convexhullarea.append( self.motilitys[i].getGlobalHullArea() )
			diffussion_fit_values.append( self.motilitys[i].getGeneralisedDiffusionCoefficientFit() )

		xsize = ( 6*len(diffussion_fit_values) ,5) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = convexhullarea, ydata = diffussion_fit_values, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit, window = window, xlabel = r'$ A $' , ylabel = r'$ K_{\beta} $', is_multi = is_multi, size = xsize, markersize = markersize, alpha = alpha, xscale = xscale, yscale = yscale, is_legend = is_legend )

	def showDiffusionCoefficientConvexHullPerimeter ( self, is_set_fit_TAMSD = False, window_TAMSD = (1,6), is_fit = False, is_fit_all = False, type_fit = 'linear', window = None, is_multi = False, markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear', is_legend = False ) :

		if is_set_fit_TAMSD :
			for i in range( len(self.motilitys) ):
				self.motilitys[i].setFitTAMSD( window = window_TAMSD )
		
		convexhullperimeter = []
		diffussion_fit_values = []
		for i in range( len(self.motilitys) ):
			convexhullperimeter.append( self.motilitys[i].getGlobalHullPerimeter() )
			diffussion_fit_values.append( self.motilitys[i].getGeneralisedDiffusionCoefficientFit() )

		xsize = ( 6*len(diffussion_fit_values) ,5) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = convexhullperimeter, ydata = diffussion_fit_values, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit, window = window, xlabel = r'$ P $' , ylabel = r'$ K_{\beta} $', is_multi = is_multi, size = xsize, markersize = markersize, alpha = alpha, xscale = xscale, yscale = yscale, is_legend = is_legend )

	def showDiffusionCoefficientTotalDistanceTraveled ( self, is_set_fit_TAMSD = False, window_TAMSD = (1,6), is_fit = False, is_fit_all = False, type_fit = 'linear', window = None, is_multi = False, markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear', is_legend = False ):

		if is_set_fit_TAMSD :
			for i in range( len(self.motilitys) ):
				self.motilitys[i].setFitTAMSD( window = window_TAMSD )
		
		totaldistancetraveled = []
		diffussion_fit_values = []
		for i in range( len(self.motilitys) ):
			totaldistancetraveled.append( self.motilitys[i].getGlobalPath() )
			diffussion_fit_values.append( self.motilitys[i].getGeneralisedDiffusionCoefficientFit() )

		xsize = ( 6*len(diffussion_fit_values) ,5) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = totaldistancetraveled, ydata = diffussion_fit_values, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit, window = window, xlabel = r' Total distance traveled ($\mu m$)' , ylabel = r'$ K_{\beta} $', is_multi = is_multi, size = xsize, markersize = markersize, alpha = alpha, xscale = xscale, yscale = yscale, is_legend = is_legend )

	def showDiffusionCoefficientNetDistanceTraveled ( self, is_set_fit_TAMSD = False, window_TAMSD = (1,6), is_fit = False, is_fit_all = False, type_fit = 'linear', window = None, is_multi = False, markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear', is_legend = False ):

		if is_set_fit_TAMSD :
			for i in range( len(self.motilitys) ):
				self.motilitys[i].setFitTAMSD( window = window_TAMSD )
		
		netdistancetraveled = []
		diffussion_fit_values = []
		for i in range( len(self.motilitys) ):
			netdistancetraveled.append( self.motilitys[i].getGlobalEndtoEndDistance() )
			diffussion_fit_values.append( self.motilitys[i].getGeneralisedDiffusionCoefficientFit() )

		xsize = ( 6*len(diffussion_fit_values) ,5) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = netdistancetraveled, ydata = diffussion_fit_values, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit, window = window, xlabel = r'Net distance traveled ($\mu m$)' , ylabel = r'$ K_{\beta} $', is_multi = is_multi, size = xsize, markersize = markersize, alpha = alpha, xscale = xscale, yscale = yscale, is_legend = is_legend )	

	def showDiffusionCoefficientMaxDistanceTraveled ( self, is_set_fit_TAMSD = False, window_TAMSD = (1,6), is_fit = False, is_fit_all = False, type_fit = 'linear', window = None, is_multi = False, markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear', is_legend = False ):

		if is_set_fit_TAMSD :
			for i in range( len(self.motilitys) ):
				self.motilitys[i].setFitTAMSD( window = window_TAMSD )
		
		maxdistance = []
		diffussion_fit_values = []
		for i in range( len(self.motilitys) ):
			maxdistance.append( self.motilitys[i].getGlobalMaxDistance() )
			diffussion_fit_values.append( self.motilitys[i].getGeneralisedDiffusionCoefficientFit() )

		xsize = ( 6*len(diffussion_fit_values) ,5) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = maxdistance, ydata = diffussion_fit_values, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit, window = window, xlabel = r'Max distance traveled ($\mu m$)' , ylabel = r'$ K_{\beta} $', is_multi = is_multi, size = xsize, markersize = markersize, alpha = alpha, xscale = xscale, yscale = yscale, is_legend = is_legend )

	# convex hull 

	def showConvexHullAreaConfinementRatio ( self, ylim = None, is_multi = False, is_fit = False, is_fit_all = False, type_fit = 'linear', window = None, markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear', is_legend = False ):

		convexhullarea = []
		ratio_confinement = []
		for i in range( len(self.motilitys) ):
			convexhullarea.append( self.motilitys[i].getGlobalHullArea() )
			ratio_confinement.append( self.motilitys[i].getGlobalPersistenceRatio() )

		xsize = ( 6*len(self.motilitys) ,5) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = ratio_confinement, ydata = convexhullarea, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit, window = window, xlabel = 'Confinement ratio' , ylabel = r'A $(\mu m^2)$', is_multi = is_multi, size = xsize, markersize = markersize, alpha = alpha, xscale = xscale, yscale = yscale, is_legend = is_legend )

	# all Moments MSD

	def showMomentRatioMSD( self, is_multi= False, is_color = False, is_legend = False, xscale = 'linear', yscale = 'linear' ):

		time = []
		ratio = []
				
		for i in range( len(self.motilitys) ):
			xtime, xratio, = self.motilitys[i].getMomentRatioMSD()	
			time.append( xtime )
			ratio.append( xratio )
		
		xsize = (6*len(self.motilitys), 5) if is_multi else (6, 5)
		xinfo_fit, xplt = self._util.scheme_scatter( xdata = time, ydata = ratio, xlabel = 't (min)', ylabel = r'$\langle r(t)^4 \rangle / {\langle r(t)^2 \rangle}^2 $', size = xsize, alpha = 0.5, is_multi = is_multi, is_color = is_color, is_legend = is_legend, xscale = xscale, yscale = yscale )
		if not is_multi:
			i_nonnan = numpy.where( ~numpy.isnan(ratio[0]) )[0]
			xplt.plot([0, time[0][i_nonnan[-1]] ],[2,2], color='black', linestyle=':')

		return xinfo_fit, xplt

	def showMomentRatioMME( self, is_multi= False, is_color = False, is_legend = False, xscale = 'linear', yscale = 'linear' ):

		time = []
		ratio = []
				
		for i in range( len(self.motilitys) ):
			xtime, xratio, = self.motilitys[i].getMomentRatioMME()	
			time.append( xtime )
			ratio.append( xratio )

		xsize = (6*len(self.motilitys), 5) if is_multi else (6, 5)
		xinfo_fit, xplt = self._util.scheme_plot_fill( xdata = time, ydata = ratio, xlabel = 't (min)', ylabel = r'$\langle r(t)_{max}^4 \rangle / {\langle r(t)_{max}^2 \rangle}^2 $', size = xsize, alpha = 0.5, is_multi = is_multi, is_color = is_color, is_legend = is_legend, xscale = xscale, yscale = yscale )
		if not is_multi:
			i_nonnan = numpy.where( ~numpy.isnan(ratio[0]) )[0]
			xplt.plot([0, time[0][i_nonnan[-1]] ],[1.49,1.49], color='black', linestyle=':')

		return xinfo_fit, xplt

	def showMomentsMSD ( self, exponents = [2], is_fit = False, window = (0,10), is_legend = False, xscale = 'linear', yscale = 'linear' ):

		def fit_tamsd(t, alpha, org):
			return alpha*t + org

		const_fits = []
		plt.figure( figsize =( 6*len(self.motilitys) ,5) )

		for i in range( len(self.motilitys) ):
			time, moments = self.motilitys[i].getMomentsMSD( scaling_exponent = exponents )	
			xcolor = list( Color(self._colors[i]).range_to( Color('black'), len(exponents) ) )
			plt.subplot(1, len(self.motilitys), i+1)
			plt.title( self._names[i], fontdict = { 'fontsize' : self._font_size } )
			for j, xmoments in enumerate(moments):
				plt.scatter( time, xmoments, marker = 'o', c = xcolor[j].hex, s = 100, alpha = 0.5, label = 'q = '+str(exponents[j]) )			
			
				if is_fit :
					xrest = curve_fit( fit_tamsd, numpy.log10(time[window[0]:window[1]]), numpy.log10( xmoments[ window[0]:window[1] ] ) )
					plt.plot( time[window[0]:window[1]],  numpy.power(time[window[0]:window[1]], xrest[0][0] )*math.pow(10, xrest[0][1]), color = xcolor[j].hex, lw = 2 )
					plt.text( numpy.average(time[0]), numpy.average(xmoments[0]), ' slope '+str( round(xrest[0][0],2) ), color = xcolor[j].hex )

					const_fits.append( { 'name':self._names[i], 'moment':exponents[j], 'constant': math.pow(10, xrest[0][1]), 'slope': xrest[0][0] } )

			plt.xscale( xscale )
			plt.yscale( yscale )
			plt.xticks( fontsize = self._font_size )
			plt.yticks( fontsize = self._font_size )
			plt.xlabel( 'time (min)', fontdict = { 'size' : self._font_size } )
			plt.ylabel( r'MSD $(\mu m^q)$', fontdict = { 'size' : self._font_size } )
			plt.grid( linestyle = ':' )
			if is_legend:
				plt.legend( frameon = False )
	
		plt.tight_layout()

		return const_fits, plt

	def showMomentScalingExponent ( self, exponents = [2], window_moment = (0,10), is_fit = False, window = None, type_fit = 'linear', is_multi = False, alpha = 0.5, markersize = 120, is_legend = False ):

		xtype_fit = [type_fit]*len(self.motilitys) if type(type_fit) is not list else type_fit
		xwindow = [window]*len(self.motilitys) if type(window) is not list else window

		const_fits = []

		xsize = (6*len(self.motilitys), 5) if is_multi else (6, 5)
		if is_multi :
			plt.figure( figsize = xsize )

		for i in range( len(self.motilitys) ):
			time, moments = self.motilitys[i].getMomentsMSD( scaling_exponent = exponents )	
			
			scaling_exponent = []
			diffusion_coefficient = []
			for j, xmoments in enumerate(moments):
				xrest = linregress( numpy.log10(time[window_moment[0]:window_moment[1]]), numpy.log10( xmoments[ window_moment[0]:window_moment[1] ] ) )
				scaling_exponent.append( xrest.slope )
				diffusion_coefficient.append( math.pow(10, xrest.intercept) )
			
			if is_multi:
				plt.subplot(1, len(self.motilitys), i+1)
				plt.title( self._names[i], fontdict = { 'fontsize' : self._font_size } )

			plt.scatter( exponents, scaling_exponent, c = self._colors[i], label = self._names[i], alpha = alpha, s = markersize, marker = 'o' )
			if is_multi:
				plt.fill_between( exponents, numpy.array(exponents)/2, exponents, color = 'lightgray', alpha = 0.5  )

			if is_fit :
				
				if xtype_fit[i] == 'linear' :
					xsplit = numpy.array(exponents)[ xwindow[i][0]:xwindow[i][1] ] if xwindow[i] else numpy.array(exponents)
					ysplit = numpy.array(scaling_exponent)[ xwindow[i][0]:xwindow[i][1] ] if xwindow[i] else numpy.array(scaling_exponent)
					res = linregress( xsplit, ysplit )

					plt.plot( xsplit, res.intercept + res.slope*xsplit, color = self._colors[i] )

					const_fits.append( { 'slope' : res.slope, 'intercept' : res.intercept } )

				elif xtype_fit[i] == 'bilinear' :
					xsplit_1 = numpy.array(exponents)[ xwindow[i][0][0]:xwindow[i][0][1] ] if xwindow[i] else numpy.array(exponents)
					ysplit_1 = numpy.array(scaling_exponent)[ xwindow[i][0][0]:xwindow[i][0][1] ] if xwindow[i] else numpy.array(scaling_exponent)

					xsplit_2 = numpy.array(exponents)[ xwindow[i][1][0]:xwindow[i][1][1] ] if xwindow[i] else numpy.array(exponents)
					ysplit_2 = numpy.array(scaling_exponent)[ xwindow[i][1][0]:xwindow[i][1][1] ] if xwindow[i] else numpy.array(scaling_exponent)
					
					res_1 = linregress( xsplit_1, ysplit_1 )
					res_2 = linregress( xsplit_2, ysplit_2 )

					x_intercept = (res_2.intercept - res_1.intercept)/(res_1.slope - res_2.slope)
					y_intercept = res_1.intercept + res_1.slope*x_intercept

					plt.plot( xsplit_1, res_1.intercept + res_1.slope*xsplit_1, color = self._colors[i] )
					plt.plot( xsplit_2, res_2.intercept + res_2.slope*xsplit_2, color = self._colors[i] )
					plt.vlines( x_intercept, 0, exponents[ len(exponents) -1 ], colors = ['darkgray'], linestyles = ':', lw = 4 )

					const_fits.append( { 'slope 1' : res_1.slope, 'intercept 1' : res_1.intercept, 'slope 2' : res_2.slope, 'intercept 2' : res_2.intercept, 'x intercept' : x_intercept, 'y intercept' : y_intercept } )
				
			if is_multi :
				plt.xticks( fontsize = self._font_size )
				plt.yticks( fontsize = self._font_size )
				plt.xlabel( 'q', fontdict = { 'size' : self._font_size } )
				plt.ylabel( r'$\zeta(q)$', fontdict = { 'size' : self._font_size } )
				plt.grid( linestyle = ':' )

		if not is_multi :
			plt.fill_between( exponents, numpy.array(exponents)/2, exponents, color = 'lightgray', alpha = 0.5  )

			plt.xticks( fontsize = self._font_size )
			plt.yticks( fontsize = self._font_size )
			plt.xlabel( 'q', fontdict = { 'size' : self._font_size } )
			plt.ylabel( r'$\zeta(q)$', fontdict = { 'size' : self._font_size } )
			plt.grid( linestyle = ':' )
			if is_legend :
				plt.legend( frameon = False )

		plt.tight_layout()
		#xsize = (6*len(self.motilitys), 5) if is_multi else (6, 5)
		#return self._util.scheme_scatter( xdata = xdata, ydata = ydata, xlabel = 'q', ylabel = r'$\zeta(q)$', size = xsize, alpha = 0.5, is_fit = is_fit, window = window, type_fit = type_fit, is_multi = is_multi, is_color = is_color, is_legend = is_legend, xscale = xscale, yscale = yscale )

		return const_fits, plt

	# gaussianity parameter

	def showGaussianityParameter ( self, is_multi= False, is_color = False, is_legend = False, xscale = 'linear', yscale = 'linear' ):
		
		time = []
		gauss = []
				
		for i in range( len(self.motilitys) ):
			xtime, xgass = self.motilitys[i].getGaussianityParameter()	
			time.append( xtime )
			gauss.append( xgass )
		
		xsize = (6*len(self.motilitys), 5) if is_multi else (6, 5)
		xinfo_fit, xplt = self._util.scheme_scatter( xdata = time, ydata = gauss, xlabel = r'$\Delta \; (min)$', ylabel = r'$G( \Delta )$', size = xsize, alpha = 0.5, is_multi = is_multi, is_color = is_color, is_legend = is_legend, xscale = xscale, yscale = yscale )
		if not is_multi:
			i_nonnan = numpy.where( ~numpy.isnan(gauss[0]) )[0]
			xplt.plot([0, time[0][i_nonnan[-1]] ],[0,0], color='black', linestyle=':')

		return xinfo_fit, xplt		

	# packing coefficient

	def showPackingCoefficient ( self, length_window = 5, is_filter = False, index_start = 0, index_end = 0, is_multi = False, is_color = True, is_legend = False ):

		time = []
		pc = []
		xpoint = []
		ypoint = []

		for i in range( len(self.motilitys) ):
			xtime, xpc, xcell, xwait_time, xcount_cell = self.motilitys[i].getPackingCoefficient( length_window = length_window )
			time.append( xtime )
			if is_filter:
				pc.append( xpc[:,index_start:index_end:] )
				tx, ptx = self.motilitys[i].getAllXPath()
				ty, pty = self.motilitys[i].getAllYPath()
				xpoint.append( ptx[ :,xcell[index_start:index_end:] ] )
				ypoint.append( pty[ :,xcell[index_start:index_end:] ] )
			else:
				pc.append( xpc )

		xsize = (6*len(self.motilitys), 5) if is_multi else (6, 5)
		if is_filter:
			
			self._util.scheme_plot_fill( xdata = time, ydata = pc, is_multi = is_multi, is_axis_equal = False, size = xsize, is_color = is_color, xlabel = ' t (min) ', ylabel = r'$P_c \; (\mu m^{-2})$', is_legend = is_legend )
			self._util.scheme_plot_fill( xdata = xpoint, ydata = ypoint, ystd = [], xlabel = r'$x \; (\mu m)$', ylabel = '$y \; (\mu m)$', size = xsize, is_multi = is_multi, is_axis_equal = True, is_color = is_color )
			return plt
		else:
			return self._util.scheme_plot_fill( xdata = time, ydata = pc, is_multi = is_multi, is_axis_equal = False, size = xsize, is_color = is_color, xlabel = ' t (min) ', ylabel = r'$P_c \; (\mu m^{-2})$', is_legend = is_legend )

	def showPackingCoefficientWithThreshold ( self, length_window = 5, threshold = None, is_multi = False, is_color = True, is_legend = False, htype = 'bar', is_fit = False, type_fit = 'normal', is_density = False, bins = 10, xscale = 'linear', yscale = 'linear' ):

		time = []
		pc = []
		wait_time = []
		count_cell = []

		for i in range( len(self.motilitys) ):
			xtime, xpc, xcell, xwait_time, xcount_cell = self.motilitys[i].getPackingCoefficient( length_window = length_window, threshold = threshold )
			time.append( xtime )
			pc.append( xpc )
			count_cell.append(xcount_cell)
			xwait_time_flat = []
			for row in xwait_time:
				for item in row:
					xwait_time_flat.append( item )
			wait_time.append( numpy.array(xwait_time_flat) )

		xsize = (6*len(self.motilitys), 5) if is_multi else (6, 5)
		if threshold:
			
			plt = self._util.scheme_plot_fill( xdata = time, ydata = pc, is_multi = is_multi, is_axis_equal = False, size = xsize, is_color = is_color, xlabel = ' t (min) ', ylabel = r'$P_c \; (\mu m^{-2})$', is_legend = is_legend )
			i_nonnan = numpy.where( ~numpy.isnan(pc[0][:,0]) )[0]
			plt.hlines( threshold, 0, time[0][ i_nonnan[-1] ] , color = 'black', linestyle = ':', lw = 2, zorder = 1000 )
			
			y_label = r'$P( t_{wait} )$' if is_density else r'$N( t_{wait} )$'
			return count_cell, self._util.scheme_hist( data = wait_time, htype = htype, is_fit = is_fit, type_fit = type_fit, ylabel = y_label, xlabel = r'$t_{wait} \; (min)$', density = is_density, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )

		else:
			return self._util.scheme_plot_fill( xdata = time, ydata = pc, is_multi = is_multi, is_axis_equal = False, size = xsize, is_color = is_color, xlabel = ' t (min) ', ylabel = r'$P_c \; (\mu m^{-2})$', is_legend = is_legend )

	# persistence time

	def showHistPersistenceTime ( self, dtype = 'hist', ylim = None, is_multi = False, marker = 'o', markersize = 10, is_fit = False, type_fit = 'normal', htype = 'bar', xscale = 'linear', yscale = 'linear', bins = 10, is_density = False, is_legend = False ):

		time = []
		for i in range( len(self.motilitys) ):
			xdata = self.motilitys[i].getDataPersistenceTime()
			time.append( xdata[:,0] )
		
		y_label = r'$P( t_p )$' if is_density else r'$N( t_p )$'
		if dtype == 'hist' :
			return self._util.scheme_hist( data = time, ylim = ylim, htype = htype, is_fit = is_fit, type_fit = type_fit, ylabel = y_label, xlabel = r'$t_p \; (min)$', density = is_density, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = time, ylim = ylim, marker = marker, markersize = markersize, is_fit = is_fit, type_fit = type_fit, ylabel = y_label, xlabel = r'$t_p \; (min)$', density = is_density, is_legend = is_legend, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )

	def showEndToEndDistanceVsTimePersistence ( self, is_multi= False, is_fit = False, is_fit_all = False, type_fit = 'linear', window = None, is_color = False, is_legend = False, alpha = 0.5, xscale = 'linear', yscale = 'linear' ) :

		time = []
		endtoend = []
				
		for i in range( len(self.motilitys) ):
			xdata = self.motilitys[i].getDataPersistenceTime()
			time.append( xdata[:,0] )
			endtoend.append( xdata[:,1] )
		
		xsize = (6*len(self.motilitys), 5) if is_multi else (6, 5)
		return self._util.scheme_scatter( xdata = time, ydata = endtoend, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit, window = window, xlabel = r'$ t_p \; (min)$', ylabel = r'$\mathcal{L}_P^{\;net} \; (\mu m) $', size = xsize, alpha = alpha, is_multi = is_multi, is_color = is_color, is_legend = is_legend, xscale = xscale, yscale = yscale )
	
	def showNetDistanceVsTimePersistence ( self, is_multi= False, is_fit = False, is_fit_all = False, type_fit = 'linear', window = None, is_color = False, is_legend = False, alpha = 0.5, xscale = 'linear', yscale = 'linear' ) :

		time = []
		distance = []
				
		for i in range( len(self.motilitys) ):
			xdata = self.motilitys[i].getDataPersistenceTime()
			time.append( xdata[:,0] )
			distance.append( xdata[:,2] )
		
		xsize = (6*len(self.motilitys), 5) if is_multi else (6, 5)
		return self._util.scheme_scatter( xdata = time, ydata = distance, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit, window = window, xlabel = r'$ t_p \; (min)$', ylabel = r'$\mathcal{L}_P^{\;total} \; (\mu m) $', size = xsize, alpha = alpha, is_multi = is_multi, is_color = is_color, is_legend = is_legend, xscale = xscale, yscale = yscale )

	# global FMI

	def showGlobalFMIx ( self, is_test = False ) :

		fmi_x = []
		for i in range( len(self.motilitys) ):
			fmi_x.append( self.motilitys[i].getGlobalXFMI( is_not_nan = True ) )

		return self._util.scheme_single_boxplot( data = fmi_x, ylabel = r'$FMI_x$', is_test = is_test )

	def showGlobalFMIy ( self, is_test = False ):
		
		fmi_y = []
		for i in range( len(self.motilitys) ):
			fmi_y.append( self.motilitys[i].getGlobalYFMI( is_not_nan = True ) )

		return self._util.scheme_single_boxplot( data = fmi_y, ylabel = r'$FMI_y$', is_test = is_test )

	def showGlobalFMI ( self, is_test = False, is_ns_test = False, showfliers = True ):

		fmi_x = []
		fmi_y = []
		for i in range( len(self.motilitys) ):
			fmi_x.append( self.motilitys[i].getGlobalXFMI( is_not_nan = True ) )
			fmi_y.append( self.motilitys[i].getGlobalYFMI( is_not_nan = True ) )

		return self._util.scheme_multiple_boxplot( data = [ fmi_x, fmi_y ], xlabels = self._names , ylabel = r'$FMI$', spancap = 3, color_box = [ 'darkorange','bisque' ], is_legend = True, label_legend = [r'$FMI_x$',r'$FMI_y$'], showfliers = showfliers, is_test = is_test, is_ns_test = is_ns_test )

	# global lengths

	def showGlobalEndtoEndDistance ( self, is_test = False ):

		endtoenddistance = []
		for i in range( len(self.motilitys) ):
			endtoenddistance.append( self.motilitys[i].getGlobalEndtoEndDistance() )

		return self._util.scheme_single_boxplot( data = endtoenddistance, ylabel = 'Net distance traveled ($\mu m$)', is_test = is_test )

	def showGlobalPath ( self, is_test = False ):

		pathtotal = []
		for i in range( len(self.motilitys) ):
			pathtotal.append( self.motilitys[i].getGlobalPath() )

		return self._util.scheme_single_boxplot( data = pathtotal, ylabel = 'Total distance traveled ($\mu m$)', is_test = is_test )

	def showGlobalBlobDiameter ( self, is_test = False ):

		blobdiameter = []
		for i in range( len(self.motilitys) ):
			blobdiameter.append( self.motilitys[i].getGlobalDiameterBlob() )

		return self._util.scheme_single_boxplot( data = blobdiameter, ylabel = 'Blob diameter ($\mu m$)', is_test = is_test )

	def showGlobalMaxDistance ( self, is_test = False ):

		maxdistance = []
		for i in range( len(self.motilitys) ):
			maxdistance.append( self.motilitys[i].getGlobalMaxDistance() )

		return self._util.scheme_single_boxplot( data = maxdistance, ylabel = 'Max distance traveled ($\mu m$)', is_test = is_test )		

	def showGlobalAllLengths ( self, is_test = False ):

		lengths = []
		for i in range( len(self.motilitys) ):
			path = self.motilitys[i].getGlobalPath()		
			endtoend = self.motilitys[i].getGlobalEndtoEndDistance()
			blobd = self.motilitys[i].getGlobalDiameterBlob()
			maxd = self.motilitys[i].getGlobalMaxDistance()

			lengths.append([ path, endtoend, blobd, maxd ])

		return self._util.scheme_multiple_boxplot( data = lengths, xlabels = ['Total distance traveled','Net distance traveled','Blob diameter','Max distance traveled'], ylabel = 'Length ($\mu m$)', is_legend = True, spancap = 3, is_test = is_test )

	def showGlobalAllLengthsWOBlobDiameter ( self, is_test = False ):

		lengths = []
		for i in range( len(self.motilitys) ):
			path = self.motilitys[i].getGlobalPath()		
			endtoend = self.motilitys[i].getGlobalEndtoEndDistance()
			maxd = self.motilitys[i].getGlobalMaxDistance()

			lengths.append([ path, endtoend, maxd ])

		return self._util.scheme_multiple_boxplot( data = lengths, xlabels = ['Total distance traveled','Net distance traveled','Max distance traveled'], ylabel = 'Length ($\mu m$)', is_legend = True, spancap = 3, is_test = is_test )

	# global ratios

	def showGlobalPersistenceRatio ( self, is_test = False ):

		persistenceratio = []
		for i in range( len(self.motilitys) ):
			persistenceratio.append( self.motilitys[i].getGlobalPersistenceRatio() )

		return self._util.scheme_single_boxplot( data = persistenceratio, ylabel = 'Persistence ratio', is_test = is_test )

	def showGlobalDirectionalityRatio ( self, is_test = False, is_ns_test = False, showfliers = True ):

		directratio = []
		for i in range( len(self.motilitys) ):
			directratio.append( self.motilitys[i].getGlobalDirectionalityRatio( is_not_nan = True ) )

		return self._util.scheme_single_boxplot( data = directratio, ylabel = 'Directionality ratio', is_test = is_test, is_ns_test = is_ns_test, showfliers = showfliers )

	def showGlobalDisplacementRatio ( self, is_test = False ):

		displacratio = []
		for i in range( len(self.motilitys) ):
			displacratio.append( self.motilitys[i].getGlobalDisplacementRatio() )

		return self._util.scheme_single_boxplot( data = displacratio, ylabel = 'Displacement ratio', is_test = is_test )

	def showGlobalOutreachRatio ( self, is_test = False ):

		outreachratio = []
		for i in range( len(self.motilitys) ):
			outreachratio.append( self.motilitys[i].getGlobalOutreachRatio() )

		return self._util.scheme_single_boxplot( data = outreachratio, ylabel = 'Outreach ratio', is_test = is_test )		

	def showGlobalExplorerRatio ( self, is_test = False ):

		explorerratio = []
		for i in range( len(self.motilitys) ):
			explorerratio.append( self.motilitys[i].getGlobalExplorerRatio() )

		return self._util.scheme_single_boxplot( data = explorerratio, ylabel = 'Explorer ratio', is_test = is_test )

	def showGlobalAllRatios ( self, is_test = False ):

		ratios = []
		for i in range( len(self.motilitys) ):
			persistenceratio = self.motilitys[i].getGlobalPersistenceRatio()		
			displacratio = self.motilitys[i].getGlobalDisplacementRatio()
			outreachratio = self.motilitys[i].getGlobalOutreachRatio()
			explorerratio = self.motilitys[i].getGlobalExplorerRatio()

			ratios.append([ persistenceratio, displacratio, outreachratio, explorerratio ])

		return self._util.scheme_multiple_boxplot( data = ratios, xlabels = ['Persistence ratio','Displacement ratio','Outreach ratio','Explorer ratio'], ylabel = '', is_legend = True, spancap = 3, is_test = is_test )

	# global speeds - Scaling exponents, Diffusion Coefficient, fractal dimension, Confinement Ratio, Displacement Ratio

	def showMeanCurvilinearSpeedScalingExponent ( self, is_set_fit_TAMSD = False, window_TAMSD = (1,6), is_fit = False, is_fit_all = False, type_fit = 'linear', window = None, is_multi = False, markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear' ) :

		if is_set_fit_TAMSD :
			for i in range( len(self.motilitys) ):
				self.motilitys[i].setFitTAMSD( window = window_TAMSD )
		
		exponent_fit_values = []
		meancurvilinearspeed = []
		for i in range( len(self.motilitys) ):
			exponent_fit_values.append( self.motilitys[i].getScalingExponentFit() )
			meancurvilinearspeed.append( self.motilitys[i].getGlobalMeanCurvilinearSpeed() )

		xsize = ( 6*len(self.motilitys) ,5) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = exponent_fit_values, ydata = meancurvilinearspeed, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit, window = window, ylabel = 'Mean Curvilinear Speed ($\mu m/min$)' , xlabel = r'$\beta$', is_multi = is_multi, size = xsize, markersize = markersize, alpha = alpha, xscale = xscale, yscale = yscale )

	def showMeanStraightLineSpeedScalingExponent ( self, is_set_fit_TAMSD = False, window_TAMSD = (1,6), is_fit = False, is_fit_all = False, type_fit = 'linear', window = None, is_multi = False, markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear' ) :

		if is_set_fit_TAMSD :
			for i in range( len(self.motilitys) ):
				self.motilitys[i].setFitTAMSD( window = window_TAMSD )
		
		exponent_fit_values = []
		meanstraightlinespeed = []
		for i in range( len(self.motilitys) ):
			exponent_fit_values.append( self.motilitys[i].getScalingExponentFit() )
			meanstraightlinespeed.append( self.motilitys[i].getGlobalMeanStraightLineSpeed() )

		xsize = ( 6*len(self.motilitys) ,5) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = exponent_fit_values, ydata = meanstraightlinespeed, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit, window = window, ylabel = 'Mean Straight Line Speed ($\mu m/min$)' , xlabel = r'$\beta$', is_multi = is_multi, size = xsize, markersize = markersize, alpha = alpha, xscale = xscale, yscale = yscale )

	def showMeanCurvilinearSpeedDiffusionCoefficient ( self, is_set_fit_TAMSD = False, window_TAMSD = (1,6), is_fit = False, is_fit_all = False, type_fit = 'linear', window = None, is_multi = False, markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear' ):

		if is_set_fit_TAMSD :
			for i in range( len(self.motilitys) ):
				self.motilitys[i].setFitTAMSD( window = window_TAMSD )
		
		meancurvilinearspeed = []
		diffussion_fit_values = []
		for i in range( len(self.motilitys) ):
			meancurvilinearspeed.append( self.motilitys[i].getGlobalMeanCurvilinearSpeed() )
			diffussion_fit_values.append( self.motilitys[i].getGeneralisedDiffusionCoefficientFit() )

		xsize = ( 6*len(self.motilitys) ,5) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = diffussion_fit_values, ydata = meancurvilinearspeed, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit, window = window, ylabel = r'Mean Curvilinear Speed ($\mu m/min$)' , xlabel = r'$ K_{\beta} $', is_multi = is_multi, size = xsize, markersize = markersize, alpha = alpha, xscale = xscale, yscale = yscale )

	def showMeanStraightLineSpeedDiffusionCoefficient ( self, is_set_fit_TAMSD = False, window_TAMSD = (1,6), is_fit = False, is_fit_all = False, type_fit = 'linear', window = None, is_multi = False, markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear' ):

		if is_set_fit_TAMSD :
			for i in range( len(self.motilitys) ):
				self.motilitys[i].setFitTAMSD( window = window_TAMSD )
		
		meanstraightlinespeed = []
		diffussion_fit_values = []
		for i in range( len(self.motilitys) ):
			meanstraightlinespeed.append( self.motilitys[i].getGlobalMeanStraightLineSpeed() )
			diffussion_fit_values.append( self.motilitys[i].getGeneralisedDiffusionCoefficientFit() )

		xsize = ( 6*len(self.motilitys) ,5) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = diffussion_fit_values, ydata = meanstraightlinespeed, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit, window = window, ylabel = r'Mean Straight Line Speed ($\mu m/min$)' , xlabel = r'$ K_{\beta} $', is_multi = is_multi, size = xsize, markersize = markersize, alpha = alpha, xscale = xscale, yscale = yscale )

	def showMeanCurvilinearSpeedFractalDimension ( self, is_fit = False, is_fit_all = False, type_fit = 'linear', window = None, is_multi = False, markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear' ) :
		
		fractal = []
		meancurvilinearspeed = []
		for i in range( len(self.motilitys) ):
			fractal.append( self.motilitys[i].getGlobalFractalDimension( ) )
			meancurvilinearspeed.append( self.motilitys[i].getGlobalMeanCurvilinearSpeed() )

		xsize = ( 6*len(self.motilitys) ,5) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = fractal, ydata = meancurvilinearspeed, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit, window = window, ylabel = 'Mean Curvilinear Speed ($\mu m/min$)' , xlabel = r'$d_f$', is_multi = is_multi, size = xsize, markersize = markersize, alpha = alpha, xscale = xscale, yscale = yscale )

	def showMeanStraightLineSpeedFractalDimension ( self, is_fit = False, is_fit_all = False, type_fit = 'linear', window = None, is_multi = False, markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear' ) :
		
		fractal = []
		meanstraightlinespeed = []
		for i in range( len(self.motilitys) ):
			fractal.append( self.motilitys[i].getGlobalFractalDimension( ) )
			meanstraightlinespeed.append( self.motilitys[i].getGlobalMeanStraightLineSpeed() )

		xsize = ( 6*len(self.motilitys) ,5) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = fractal, ydata = meanstraightlinespeed, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit, window = window, ylabel = 'Mean Straight Line Speed ($\mu m/min$)' , xlabel = r'$d_f$', is_multi = is_multi, size = xsize, markersize = markersize, alpha = alpha, xscale = xscale, yscale = yscale )

	def showMeanCurvilinearSpeedConfinementRatio ( self, is_fit = False, is_fit_all = False, type_fit = 'linear', window = None, is_multi = False, markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear' ) :
		
		ratio_confinement = []
		meancurvilinearspeed = []
		for i in range( len(self.motilitys) ):
			ratio_confinement.append( self.motilitys[i].getGlobalPersistenceRatio() )
			meancurvilinearspeed.append( self.motilitys[i].getGlobalMeanCurvilinearSpeed() )

		xsize = ( 6*len(self.motilitys) ,5) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = ratio_confinement, ydata = meancurvilinearspeed, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit, window = window, ylabel = 'Mean Curvilinear Speed ($\mu m/min$)' , xlabel = 'Confinement ratio', is_multi = is_multi, size = xsize, markersize = markersize, alpha = alpha, xscale = xscale, yscale = yscale )

	def showMeanStraightLineSpeedConfinementRatio ( self, is_fit = False, is_fit_all = False, type_fit = 'linear', window = None, is_multi = False, markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear' ) :
		
		ratio_confinement = []
		meanstraightlinespeed = []
		for i in range( len(self.motilitys) ):
			ratio_confinement.append( self.motilitys[i].getGlobalPersistenceRatio() )
			meanstraightlinespeed.append( self.motilitys[i].getGlobalMeanStraightLineSpeed() )

		xsize = ( 6*len(self.motilitys) ,5) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = ratio_confinement, ydata = meanstraightlinespeed, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit, window = window, ylabel = 'Mean Straight Line Speed ($\mu m/min$)' , xlabel = 'Confinement ratio', is_multi = is_multi, size = xsize, markersize = markersize, alpha = alpha, xscale = xscale, yscale = yscale )

	def showMeanCurvilinearSpeedDisplacementRatio ( self, is_fit = False, is_fit_all = False, type_fit = 'linear', window = None, is_multi = False, markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear' ) :
		
		ratio_displacement = []
		meancurvilinearspeed = []
		for i in range( len(self.motilitys) ):
			ratio_displacement.append( self.motilitys[i].getGlobalDisplacementRatio() )
			meancurvilinearspeed.append( self.motilitys[i].getGlobalMeanCurvilinearSpeed() )

		xsize = ( 6*len(self.motilitys) ,5) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = ratio_displacement, ydata = meancurvilinearspeed, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit, window = window, ylabel = 'Mean Curvilinear Speed ($\mu m/min$)' , xlabel = 'Displacement ratio', is_multi = is_multi, size = xsize, markersize = markersize, alpha = alpha, xscale = xscale, yscale = yscale )

	def showMeanStraightLineSpeedDisplacementRatio ( self, is_fit = False, is_fit_all = False, type_fit = 'linear', window = None, is_multi = False, markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear' ) :
		
		ratio_displacement = []
		meanstraightlinespeed = []
		for i in range( len(self.motilitys) ):
			ratio_displacement.append( self.motilitys[i].getGlobalDisplacementRatio() )
			meanstraightlinespeed.append( self.motilitys[i].getGlobalMeanStraightLineSpeed() )

		xsize = ( 6*len(self.motilitys) ,5) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = ratio_displacement, ydata = meanstraightlinespeed, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit, window = window, ylabel = 'Mean Straight Line Speed ($\mu m/min$)' , xlabel = 'Displacement ratio', is_multi = is_multi, size = xsize, markersize = markersize, alpha = alpha, xscale = xscale, yscale = yscale )

	# global speeds

	def showGlobalMeanStraightLineSpeed ( self, is_test = False ):

		avgspeed = []
		for i in range( len(self.motilitys) ):
			avgspeed.append( self.motilitys[i].getGlobalMeanStraightLineSpeed() )

		return self._util.scheme_single_boxplot( data = avgspeed, ylabel = 'Mean Straight Line Speed ($\mu m / min$)', is_test = is_test )

	def showGlobalXMeanStraightLineSpeed ( self, is_test = False ):
		
		xavgspeed = []
		for i in range( len(self.motilitys) ):
			xavgspeed.append( self.motilitys[i].getGlobalXMeanStraightLineSpeed() )

		return self._util.scheme_single_boxplot( data = xavgspeed, ylabel = r'$ \nu_x \; (\mu m / min)$', is_test = is_test )

	def showGlobalYMeanStraightLineSpeed ( self, is_test = False ):
		
		yavgspeed = []
		for i in range( len(self.motilitys) ):
			yavgspeed.append( self.motilitys[i].getGlobalYMeanStraightLineSpeed() )

		return self._util.scheme_single_boxplot( data = yavgspeed, ylabel = r'$ \nu_y \; (\mu m / min)$', is_test = is_test )

	def showGlobalXYMeanStraightLineSpeed ( self, is_test = False, is_ns_test = False, showfliers = True ):

		xspeed = []
		yspeed = []
		for i in range( len(self.motilitys) ):
			xspeed.append( self.motilitys[i].getGlobalXMeanStraightLineSpeed() )
			yspeed.append( self.motilitys[i].getGlobalYMeanStraightLineSpeed() )

		return self._util.scheme_multiple_boxplot( data = [ xspeed, yspeed ], xlabels = self._names , ylabel = 'Mean Straight Line Speed ($\mu m/min$)', is_legend = True, color_box = [ 'maroon', 'red' ], label_legend = [ r'$\nu_x$', r'$\nu_y$' ], showfliers = showfliers, spancap = 3, is_test = is_test, is_ns_test = is_ns_test )

	def showGlobalTotalSpeed ( self, is_test = False ) :

		totalspeed = []
		for i in range( len(self.motilitys) ):
			totalspeed.append( self.motilitys[i].getGlobalTotalSpeed() )
		
		return self._util.scheme_single_boxplot( data = totalspeed, ylabel = 'Total Speed ($\mu m / min$)', is_test = is_test )

	def showGlobalMeanCurvilinearSpeed ( self, is_test = False ):

		avgcurvspeed = []
		for i in range( len(self.motilitys) ):
			avgcurvspeed.append( self.motilitys[i].getGlobalMeanCurvilinearSpeed() )

		return self._util.scheme_single_boxplot( data = avgcurvspeed, ylabel = 'Mean Curvilinear Speed ($\mu m / min$)', is_test = is_test )

	def showGlobalGSpeed ( self, data_vg = [] ):

		for i in range( len(self.motilitys) ):
			self.motilitys[i].computeGSpeed( explore_fit = data_vg[i] )

		gspeed = []
		stdgspeed = []
		index_position = []
		for i in range( len(self.motilitys) ):
			gspeed.append( self.motilitys[i].getGSpeed() )
			stdgspeed.append( self.motilitys[i].getStdGSpeed() )
			index_position.append( i + 1 )

		plt.bar( x = index_position, height = gspeed, yerr = stdgspeed, color = self._colors )
		plt.ylabel(r'$V_{g} \; (\mu m.min^{-1})$',fontdict = {'size' : self._font_size})
		plt.xticks( index_position, self._names, fontsize = self._font_size, rotation = 45 )
		plt.yticks( fontsize = self._font_size )
		plt.grid( linestyle = ':' )

		return gspeed, stdgspeed, plt

	# global view angle

	def showGlobalHistMaxViewAngle ( self, is_multi = False, htype = 'bar' ):

		angle = []
		for i in range( len(self.motilitys) ):
			angle.append( self.motilitys[i].getGlobalMaxViewAngle() )
		
		return self._util.scheme_hist( data = angle, ylabel = 'PDF', xlabel = 'Angle (deg)', htype = htype, density = True, is_legend = True, is_multi = is_multi )

	def showGlobalMaxViewAngle ( self, is_test = False ):
		
		angle = []
		for i in range( len(self.motilitys) ):
			angle.append( self.motilitys[i].getGlobalMaxViewAngle() )
		
		return self._util.scheme_single_boxplot( data = angle, ylabel = 'Angle (deg)', is_test = is_test )


	def showGlobalHistTrackingViewAngle ( self, is_multi = False, htype = 'bar' ):

		angle = []
		for i in range( len(self.motilitys) ):
			angle.append( self.motilitys[i].getGlobalTrackingViewAngle() )
		
		return self._util.scheme_hist( data = angle, ylabel = 'PDF', xlabel = 'Angle (deg)', htype = htype, density = True, is_legend = True, is_multi = is_multi )

	def showGlobalTrackingViewAngle ( self, is_test = False ):

		angle = []
		for i in range( len(self.motilitys) ):
			angle.append( self.motilitys[i].getGlobalTrackingViewAngle() )
		
		return self._util.scheme_single_boxplot( data = angle, ylabel = 'Angle (deg)', is_test = is_test )

	# global orientation angle

	def showGlobalOrientationAngle ( self, size = (12,4), bin_size = 20 ):
		angle = []
		for i in range( len(self.motilitys) ):
			angle.append( self.motilitys[i].getGlobalMassCenterOrientation() )

		return self._util.scheme_polar_histogram( data = angle, size = size, bin_size = bin_size )


	# detail orientation angle

	def showAllOrientationAngle ( self, size = (12,4), bin_size = 20 ):
		angle = []
		for i in range( len(self.motilitys) ):
			angle.append( self.motilitys[i].getAllAngleOrientation( is_flat = True ) )

		return self._util.scheme_polar_histogram( data = angle, size = size, bin_size = bin_size )

	
	def showAngleOrientationTime ( self, is_multi = False, is_color = False, is_legend = False ):
		angle = []
		time = []
		for i in range( len(self.motilitys) ):
			xtime, xangle = self.motilitys[i].getAngleOrientationTime( is_shift_zero = True  )
			angle.append( xangle )
			time.append( xtime )

		xsize = (6*len(self.motilitys),5) if is_multi else (6,5)
		return self._util.scheme_plot_fill( xdata = time, ydata = angle, xlabel = 'time (min)', ylabel = 'angle orientation (deg)' , is_multi = is_multi, size = xsize, is_color = is_color, is_legend = is_legend )

	# Global first orientation body cell
	def showGlobalFirstOrientationCellBody ( self, size = (12,4), bin_size = 20 ):

		angle = []
		for i in range( len(self.motilitys) ):
			angle.append( self.motilitys[i].getGlobalFirstOrientationCellBody() )

		return self._util.scheme_polar_histogram( data = angle, size = size, bin_size = bin_size )


