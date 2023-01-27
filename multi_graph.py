from motility import *
from util import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math
import numpy
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
from colour import Color

class multi_graph :

	def __init__ ( self, data = [], names = [], colors = [], micron_px = 1, time_seq = 1, time_acquisition = 0 ):

		self.motilitys = []
		self._font_size = 16
		self._names = names
		self._colors= colors
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

	# tracking
	
	def showCellXTimeTracking ( self, is_color = False ):

		xtime = []
		xpoint = []
		for i in range( len(self.motilitys) ):
			time, point = self.motilitys[i].getAllXPath()
			xtime.append( time )
			xpoint.append( point )

		return self._util.scheme_plot_fill( xdata = xtime, ydata = xpoint, ystd = [], xlabel = 't (min)', ylabel = '$x \; (\mu m)$', size = (6*len(xpoint),5), is_multi = True, is_color = is_color )

	def showCellYTimeTracking ( self, is_color = False ):

		ytime = []
		ypoint = []
		for i in range( len(self.motilitys) ):
			time, point = self.motilitys[i].getAllYPath()
			ytime.append( time )
			ypoint.append( point )

		return self._util.scheme_plot_fill( xdata = ytime, ydata = ypoint, ystd = [], xlabel = 't (min)', ylabel = '$y \; (\mu m)$', size = (6*len(ypoint),5), is_multi = True, is_color = is_color )

	def showCellXYTracking ( self, is_color = False ):

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

		return self._util.scheme_plot_fill( xdata = xpoint, ydata = ypoint, ystd = [], xlabel = r'$x \; (\mu m)$', ylabel = '$y \; (\mu m)$', size = (6*len(xpoint),5), is_multi = True, is_axis_equal = True, is_color = is_color )
	
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

	def showCellDifferenceXTimeTracking ( self, is_color = False ):

		xtime = []
		xpoint = []
		for i in range( len(self.motilitys) ):
			time, point = self.motilitys[i].getAllDeltaX()
			xtime.append( time )
			xpoint.append( point )

		return self._util.scheme_plot_fill( xdata = xtime, ydata = xpoint, ystd = [], xlabel = 't (min)', ylabel = '$\Delta x \; (\mu m)$', size = (6*len(xpoint),5), is_multi = True, is_color = is_color )

	def showCellDifferenceYTimeTracking (self, is_color = False ):

		ytime = []
		ypoint = []
		for i in range( len(self.motilitys) ):
			time, point = self.motilitys[i].getAllDeltaY()
			ytime.append( time )
			ypoint.append( point )

		return self._util.scheme_plot_fill( xdata = ytime, ydata = ypoint, ystd = [], xlabel = 't (min)', ylabel = '$\Delta y \; (\mu m)$', size = (6*len(ypoint),5), is_multi = True, is_color = is_color )

	# quartile X, Y displacement

	def showQuantileLinesDifferenceXTimeTracking ( self, quantile = [0.2,0.3,0.4,0.5,0.6,0.7,0.8] ):
		
		xtime = []
		xpoint = []
		for i in range( len(self.motilitys) ):
			time, point = self.motilitys[i].getAllDeltaX()
			xtime.append( time )
			xpoint.append( numpy.nanquantile(point, quantile, axis = 1).T )

		return self._util.scheme_plot_fill( xdata = xtime, ydata = xpoint, ystd = [], xlabel = 't (min)', ylabel = 'Quantile', size = (6*len(xpoint),5), is_multi = True, is_color = True )

	def showQuantileLinesDifferenceYTimeTracking ( self, quantile = [0.2,0.3,0.4,0.5,0.6,0.7,0.8] ):

		ytime = []
		ypoint = []
		for i in range( len(self.motilitys) ):
			time, point = self.motilitys[i].getAllDeltaY()
			ytime.append( time )
			ypoint.append( numpy.nanquantile(point, quantile, axis = 1).T )

		return self._util.scheme_plot_fill( xdata = ytime, ydata = ypoint, ystd = [], xlabel = 't (min)', ylabel = 'Quantile', size = (6*len(ypoint),5), is_multi = True, is_color = True )

	def showHistDifferenceX ( self, dtype = 'hist', is_multi = False, marker = 'o', markersize = 10, is_fit = False, type_fit = 'linear', htype = 'bar', xscale = 'linear', yscale = 'linear', ylabel = 'Count', bins = 10, is_density = False ):

		vanHoveX = []
		for i in range( len(self.motilitys) ):
			vanHoveX.append( self.motilitys[i].getAllDeltaX( is_flat= True ) )
		
		if dtype == 'hist' :
			return self._util.scheme_hist( data = vanHoveX, htype = htype, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$\Delta x \; (\mu m)$', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = vanHoveX, marker = marker, markersize = markersize, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$\Delta x \; (\mu m)$', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )

	def showHistDifferenceY ( self, dtype = 'hist', is_multi = False, marker = 'o', markersize = 10, is_fit = False, type_fit = 'linear', htype = 'bar', xscale = 'linear', yscale = 'linear', ylabel = 'Count', bins = 10, is_density = False ):

		vanHoveY = []
		for i in range( len(self.motilitys) ):
			vanHoveY.append( self.motilitys[i].getAllDeltaY( is_flat= True ) )
		
		if dtype == 'hist':
			return self._util.scheme_hist( data = vanHoveY, htype = htype, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$\Delta x \; (\mu m)$', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = vanHoveY, marker = marker, markersize = markersize, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$\Delta x \; (\mu m)$', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )

	def showHistAbsoluteDifferenceX ( self, dtype = 'hist', is_multi = False, marker = 'o', markersize = 10, is_fit = False, type_fit = 'linear', htype = 'bar', xscale = 'linear', yscale = 'linear', ylabel = 'Count', bins = 10, is_density = False ):
		
		AbsX = []
		for i in range( len(self.motilitys) ):
			AbsX.append( numpy.absolute( self.motilitys[i].getAllDeltaX( is_flat= True ) ) )
		
		if dtype == 'hist' :
			return self._util.scheme_hist( data = AbsX, htype = htype, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$|\Delta x| \; (\mu m)$', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = AbsX, marker = marker, markersize = markersize, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$|\Delta x| \; (\mu m)$', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )		

	def showHistAbsoluteDifferenceY ( self, dtype = 'hist', is_multi = False, marker = 'o', markersize = 10, is_fit = False, type_fit = 'linear', htype = 'bar', xscale = 'linear', yscale = 'linear', ylabel = 'Count', bins = 10, is_density = False ):
		
		AbsY = []
		for i in range( len(self.motilitys) ):
			AbsY.append( numpy.absolute( self.motilitys[i].getAllDeltaY( is_flat= True ) ) )
		
		if dtype == 'hist' :
			return self._util.scheme_hist( data = AbsY, htype = htype, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$|\Delta y| \; (\mu m)$', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = AbsY, marker = marker, markersize = markersize, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$|\Delta y| \; (\mu m)$', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
	
	def showHistSquaredDifferenceX ( self, dtype = 'hist', is_multi = False, marker = 'o', markersize = 10, is_fit = False, type_fit = 'linear', htype = 'bar', xscale = 'linear', yscale = 'linear', ylabel = 'Count', bins = 10, is_density = False ):

		SquaredX = []
		for i in range( len(self.motilitys) ):
			SquaredX.append( numpy.power( self.motilitys[i].getAllDeltaX( is_flat= True ),2 ) )
		
		if dtype == 'hist' :
			return self._util.scheme_hist( data = SquaredX, htype = htype, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$\Delta x^2 \; (\mu m)$', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = SquaredX, marker = marker, markersize = markersize, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$\Delta x^2 \; (\mu m)$', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )

	def showHistSquaredDifferenceY ( self, dtype = 'hist', is_multi = False, marker = 'o', markersize = 10, is_fit = False, type_fit = 'linear', htype = 'bar', xscale = 'linear', yscale = 'linear', ylabel = 'Count', bins = 10, is_density = False ):

		SquaredY = []
		for i in range( len(self.motilitys) ):
			SquaredY.append( numpy.power( self.motilitys[i].getAllDeltaY( is_flat= True ),2 ) )
		
		if dtype == 'hist' :
			return self._util.scheme_hist( data = SquaredY, htype = htype, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$\Delta y^2 \; (\mu m)$', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = SquaredY, marker = marker, markersize = markersize, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$\Delta y^2 \; (\mu m)$', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )

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
		
	def showJosephEffectXTimeTracking ( self, is_fit = False, type_fit = 'linear', window = (1,10), is_multi = False, is_legend = False, marker = 'o', markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear' ):

		xtime = []
		xpoint = []
		for i in range( len(self.motilitys) ):
			time, point = self.motilitys[i].getAllDeltaX()
			Z = numpy.power(point,2)
			R = numpy.full( point.shape , numpy.nan )
			for j in range( point.shape[0] ):
				xtemp = numpy.full( (j+1,point.shape[1]) , numpy.nan )
				for z in range(j+1):
					xtemp[z,:] = point[z,:] - point[j,:]*z/j

				R[j,:] = numpy.nanmax( xtemp, axis = 0 ) - numpy.nanmin(xtemp, axis = 0)

			S = numpy.sqrt( (Z.T/time).T - numpy.power( (point.T/time).T ,2) )

			xtime.append( time )
			xpoint.append( numpy.nanmean( R/S , axis = 1 ) )

		xsize = (6*len(xtime),len(xtime)) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = xtime, ydata = xpoint, xlabel = 't (min)', ylabel = r'$E[R_t/S_t]$', is_fit = is_fit, type_fit = type_fit, window = window, size = xsize, is_multi = is_multi, is_legend = is_legend, marker = marker, markersize = markersize, alpha = alpha, xscale = xscale , yscale = yscale )
	
	def showMosesEffectYTimeTracking (self, is_fit = False, type_fit = 'linear', window = (1,10), is_multi = False, is_legend = False, marker = 'o', markersize = 100, alpha = 0.2, xscale = 'linear', yscale = 'linear' ):
		
		xtime = []
		xpoint = []
		for i in range( len(self.motilitys) ):
			time, point = self.motilitys[i].getAllDeltaY()

			xdata = numpy.nanmedian( numpy.cumsum( numpy.absolute(point), axis = 0), axis = 1 )
			xtime.append( time[1::] )
			xpoint.append( xdata[:-1:] )

		xsize = (6*len(xtime),len(xtime)) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = xtime, ydata = xpoint, xlabel = 't (min)', ylabel = r'$m[Y_t]$', is_fit = is_fit, type_fit = type_fit, window = window, size = xsize, is_multi = is_multi, is_legend = is_legend, marker = marker, markersize = markersize, alpha = alpha, xscale = xscale , yscale = yscale )
	
	def showNoahEffectYTimeTracking ( self, is_fit = False, type_fit = 'linear', window = (1,10), is_multi = False, is_legend = False, marker = 'o', markersize = 100, alpha = 0.2, xscale = 'linear', yscale = 'linear'):

		xtime = []
		xpoint = []
		for i in range( len(self.motilitys) ):
			time, point = self.motilitys[i].getAllDeltaY()

			xdata = numpy.nanmedian( numpy.cumsum( numpy.power(point,2), axis = 0), axis = 1 )
			xtime.append( time[1::] )
			xpoint.append( xdata[:-1:] )

		xsize = (6*len(xtime),len(xtime)) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = xtime, ydata = xpoint, xlabel = 't (min)', ylabel = r'$m[Z_t]$', is_fit = is_fit, type_fit = type_fit, window = window, size = xsize, is_multi = is_multi, is_legend = is_legend, marker = marker, markersize = markersize, alpha = alpha, xscale = xscale , yscale = yscale )

	def showJosephEffectYTimeTracking ( self, is_fit = False, type_fit = 'linear', window = (1,10), is_multi = False, is_legend = False, marker = 'o', markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear' ):

		xtime = []
		xpoint = []
		for i in range( len(self.motilitys) ):
			time, point = self.motilitys[i].getAllDeltaY()
			Z = numpy.power(point,2)
			R = numpy.full( point.shape , numpy.nan )
			for j in range( point.shape[0] ):
				xtemp = numpy.full( (j+1,point.shape[1]) , numpy.nan )
				for z in range(j+1):
					xtemp[z,:] = point[z,:] - point[j,:]*z/j

				R[j,:] = numpy.nanmax( xtemp, axis = 0 ) - numpy.nanmin(xtemp, axis = 0)

			S = numpy.sqrt( (Z.T/time).T - numpy.power( (point.T/time).T ,2) )

			xtime.append( time )
			xpoint.append( numpy.nanmean( R/S , axis = 1 ) )

		xsize = (6*len(xtime),len(xtime)) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = xtime, ydata = xpoint, xlabel = 't (min)', ylabel = r'$E[R_t/S_t]$', is_fit = is_fit, type_fit = type_fit, window = window, size = xsize, is_multi = is_multi, is_legend = is_legend, marker = marker, markersize = markersize, alpha = alpha, xscale = xscale , yscale = yscale )
	
	# power spectral density (PSD) of displacement

	def showAllAndAverageXPSD ( self, is_fit = False, window = (0,20), bins = 50, xscale = 'linear', yscale = 'linear' ):

		def fit_psd(f, slope, const):
			return slope*f + const

		const_fits = []

		plt.figure( figsize = ( 8*len(self.motilitys), 5 ) )
		for i in range( len(self.motilitys) ):
			time, point = self.motilitys[i].getAllXPSD()

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
			plt.ylabel( r'PSD, <PSD> $( \mu m^2 / Hz )$', fontdict = { 'size' : self._font_size } )
			plt.grid( linestyle = ':' )

		plt.tight_layout()

		return const_fits, plt

	def showAllAndAverageYPSD ( self, is_fit = False, window = (0,20), bins = 50, xscale = 'linear', yscale = 'linear' ):

		def fit_psd(f, slope, const):
			return slope*f + const

		const_fits = []

		plt.figure( figsize = ( 8*len(self.motilitys), 5 ) )
		for i in range( len(self.motilitys) ):
			time, point = self.motilitys[i].getAllYPSD()

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
			plt.ylabel( r'PSD, <PSD> $( \mu m^2 / Hz )$', fontdict = { 'size' : self._font_size } )
			plt.grid( linestyle = ':' )

		plt.tight_layout()

		return const_fits, plt
	
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

	def showStepEndAspectRatio ( self, is_legend = False, is_fit = False, type_fit = 'linear', xscale = 'linear', yscale = 'linear' ):

		steps = []
		aspect_ratio = []
		for i in range( len(self.motilitys) ):
			xsteps = (self.motilitys[i].getGlobalLengthPoints() - 1)
			xaspect = self.motilitys[i].getGlobalEndAspectRatio()
			steps.append( xsteps )
			aspect_ratio.append( xaspect )

		self._util.scheme_scatter( is_multi = True, size = (6*len(self.motilitys),5), xdata = aspect_ratio, ydata = steps, xlabel = r'$Aspect \; ratio ( t = \Delta_f )$', ylabel = 'Steps', is_legend = is_legend, xscale = xscale, yscale = yscale, is_fit = is_fit, type_fit = type_fit )

	# detail speed

	def showScatterAllXYSpeed ( self, is_color = False, marker = 'o', markersize = 12, alpha = 0.2 ):

		xspeed = []
		yspeed = []
		for i in range( len(self.motilitys) ):
			xsp = self.motilitys[i].getAllVelocityX()
			ysp = self.motilitys[i].getAllVelocityY()
			xspeed.append(xsp)
			yspeed.append(ysp)
		
		return self._util.scheme_scatter( xdata = xspeed, ydata = yspeed, xlabel = r'$\nu_{x} \; (\mu m.min^{-1})$', ylabel = r'$\nu_{y} \; (\mu m.min^{-1})$', size = (6*len(xspeed),len(xspeed)), is_multi = True, is_color = is_color, is_legend = True, is_axis_equal = True, marker = marker, markersize = markersize, alpha = alpha )

	def showHistAllSpeed ( self, is_multi = False, htype = 'bar', dtype = 'hist', is_fit = False, type_fit = 'linear', marker = 'o', markersize = 12, xscale = 'linear', yscale = 'linear', ylabel = 'Count', bins = 10, is_density = False ):

		speed = []
		for i in range( len(self.motilitys) ):
			speed.append( self.motilitys[i].getAllVelocity( is_flat= True ) )
		
		if dtype == 'hist':
			return self._util.scheme_hist( data = speed, ylabel = ylabel, xlabel = r'$\nu \; (\mu m.min^{-1})$', htype = htype, density = is_density, is_fit = is_fit, type_fit = type_fit, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )	
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = speed, marker = marker, markersize = markersize, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$\nu \; (\mu m.min^{-1})$', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )

	def showHistAllSquaredSpeed ( self, is_multi = False, htype = 'bar', dtype = 'hist', is_fit = False, type_fit = 'linear', marker = 'o', markersize = 12, xscale = 'linear', yscale = 'linear', ylabel = 'Count', bins = 10, is_density = False ):

		squared_speed = []
		for i in range( len(self.motilitys) ):
			squared_speed.append( numpy.power( self.motilitys[i].getAllVelocity( is_flat= True ),2 ) )
		
		if dtype == 'hist':
			return self._util.scheme_hist( data = squared_speed, ylabel = ylabel, xlabel = r'$\nu^2 \; (\mu m.min^{-1})$', htype = htype, density = is_density, is_fit = True, type_fit = type_fit, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = squared_speed, marker = marker, markersize = markersize, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$\nu^2 \; (\mu m.min^{-1})$', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )

	def showAverageNormVelocityAutocorrelation ( self, show_std = False, size = (12,5), xscale = 'linear', yscale = 'linear', is_legend = True ) :
		timeVelAuto = []
		avgVelAuto = []
		stdVelAuto = []
		for i in range( len(self.motilitys) ):
			time, average, std = self.motilitys[i].getAverageVelocityCorrelation()		
			timeVelAuto.append( time )
			avgVelAuto.append( average )
			if show_std :
				stdVelAuto.append( std )

		return self._util.scheme_plot_fill( xdata = timeVelAuto, ydata = avgVelAuto, ystd = stdVelAuto, xlabel = 'Time (min)', ylabel = 'Velocity Autocorrelation', is_legend = is_legend, xscale = xscale, yscale = yscale, size = size )

	def showAveragePowerSpectrumNormVelocityAutocorrelation ( self, show_std = False, size = (12,5), xscale = 'linear', yscale = 'linear', is_legend = True ):
		
		time = []
		avg = []
		std = []
		for i in range( len(self.motilitys) ):
			xtime, xavg, xstd = self.motilitys[i].getAverageRealPowerSpectrum()
			time.append( xtime )
			avg.append( xavg )
			if show_std :
				std.append( xstd )

	
	# all MSD, TAMSD, MMA, TAMMA

	def showErgodicityBreakingParameterTAMSD ( self, is_multi= False, is_color = False, is_legend = False, xscale = 'linear', yscale = 'linear' ) :

		time = []
		eb = []

		for i in range( len(self.motilitys) ):
			xtime, xeb, = self.motilitys[i].getErgodicityBreakingParameterTAMSD()
			i_nonnan = numpy.where( ~numpy.isnan(xeb) )[0]
			time.append( xtime/i_nonnan[-1] )
			eb.append( xeb )

		xsize = (6*len(self.motilitys), 5) if is_multi else (6, 5)
		xinfo_fit, xplt = self._util.scheme_scatter( xdata = time, ydata = eb, xlabel = r'$\Delta/T$', ylabel = 'EB', size = xsize, alpha = 0.5, is_multi = is_multi, is_color = is_color, is_legend = is_legend, xscale = xscale, yscale = yscale )
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

	def showAverageTAMSD ( self, show_std = False, size = (12,5), window = (0,0), is_fit = 0, xscale = 'linear', yscale = 'linear' ) :
		timemsd = []
		avgmsd = []
		stdmsd = []
		
		def fit_msd(t, alpha, org):
			return alpha*t + org

		for i in range( len(self.motilitys) ):
			time, average, std = self.motilitys[i].getAverageTAMSD()	
			timemsd.append( time )
			avgmsd.append( average )
			if show_std :
				stdmsd.append( std )

		xplt = self._util.scheme_plot_fill( xdata = timemsd, ydata = avgmsd, ystd = stdmsd, xlabel = 'Time (min)', ylabel = '<TAMSD>', size = size, xscale = xscale, yscale = yscale, is_legend = True )
		
		const_fits = []
		if is_fit and xscale == 'log' and yscale == 'log' :
			for i in range( len(timemsd) ):
				split_time = timemsd[i][ window[i][0]:window[i][1] ] if type(window) is list else timemsd[i][ window[0]:window[1] ]
				split_average = avgmsd[i][ window[i][0]:window[i][1] ] if type(window) is list else avgmsd[i][ window[0]:window[1] ]
				xrest = curve_fit( fit_msd, numpy.log10(split_time), numpy.log10(split_average) )
				xplt.plot( split_time,  numpy.power(split_time, xrest[0][0] )*math.pow(10, xrest[0][1]), linestyle = ':', color = self._colors[i], alpha = 0.5, lw = 2 )
				xplt.text( numpy.average(split_time), numpy.average(split_average), ' slope '+str( round(xrest[0][0],2) ), color = self._colors[i] )
				const_fits.append( ( math.pow(10, xrest[0][1]), xrest[0][0] ) )

		return xplt, const_fits

	def showMSD ( self, show_std = False, size = (12,5), window = (0,0), is_fit = 0, xscale = 'linear', yscale = 'linear' ) :
		timemsd = []
		avgmsd = []
		stdmsd = []
		
		def fit_msd(t, alpha, org):
			return alpha*t + org

		for i in range( len(self.motilitys) ):
			time, average, std = self.motilitys[i].getMSD()	
			timemsd.append( time )
			avgmsd.append( average )
			if show_std :
				stdmsd.append( std )

		xplt = self._util.scheme_plot_fill( xdata = timemsd, ydata = avgmsd, ystd = stdmsd, xlabel = 'Time (min)', ylabel = 'MSD', size = size, xscale = xscale, yscale = yscale, is_legend = True )
		
		const_fits = []
		if is_fit and xscale == 'log' and yscale == 'log' :
			for i in range( len(timemsd) ):
				split_time = timemsd[i][ window[i][0]:window[i][1] ] if type(window) is list else timemsd[i][ window[0]:window[1] ]
				split_average = avgmsd[i][ window[i][0]:window[i][1] ] if type(window) is list else avgmsd[i][ window[0]:window[1] ]
				xrest = curve_fit( fit_msd, numpy.log10(split_time), numpy.log10(split_average) )
				xplt.plot( split_time,  numpy.power(split_time, xrest[0][0] )*math.pow(10, xrest[0][1]), linestyle = ':', color = self._colors[i], alpha = 0.5, lw = 2 )
				xplt.text( numpy.average(split_time), numpy.average(split_average), ' slope '+str( round(xrest[0][0],2) ), color = self._colors[i] )
				const_fits.append( ( math.pow(10, xrest[0][1]), xrest[0][0] ) )

		return xplt, const_fits

	def showMME ( self, show_std = False, size = (12,5), window = (0,0), is_fit = 0, xscale = 'linear', yscale = 'linear' ) :
		timemme = []
		avgmme = []
		stdmme = []
		
		def fit_mme(t, alpha, org):
			return alpha*t + org

		for i in range( len(self.motilitys) ):
			time, average, std = self.motilitys[i].getMME()	
			timemme.append( time )
			avgmme.append( average )
			if show_std :
				stdmme.append( std )

		xplt = self._util.scheme_plot_fill( xdata = timemme, ydata = avgmme, ystd = stdmme, xlabel = 'Time (min)', ylabel = 'MME', size = size, xscale = xscale, yscale = yscale, is_legend = True )
		
		const_fits = []
		if is_fit and xscale == 'log' and yscale == 'log' :
			for i in range( len(timemme) ):
				split_time = timemme[i][ window[i][0]:window[i][1] ] if type(window) is list else timemme[i][ window[0]:window[1] ]
				split_average = avgmme[i][ window[i][0]:window[i][1] ] if type(window) is list else avgmme[i][ window[0]:window[1] ]
				xrest = curve_fit( fit_mme, numpy.log10(split_time), numpy.log10(split_average) )
				xplt.plot( split_time,  numpy.power(split_time, xrest[0][0] )*math.pow(10, xrest[0][1]), linestyle = ':', color = self._colors[i], alpha = 0.5, lw = 2 )
				xplt.text( numpy.average(split_time), numpy.average(split_average), ' slope '+str( round(xrest[0][0],2) ), color = self._colors[i] )
				const_fits.append( ( math.pow(10, xrest[0][1]), xrest[0][0] ) )

		return xplt, const_fits

	def showHistAmplitudeTAMSD ( self, dtype = 'hist', is_multi = False, marker = 'o', markersize = 10, is_fit = False, type_fit = 'normal', htype = 'bar', xscale = 'linear', yscale = 'linear', ylabel = 'Count', bins = 10, is_density = False ):

		amplitude = []
		for i in range( len(self.motilitys) ):
			amplitude.append( self.motilitys[i].getAmplitudeTAMSD( is_flat_not_nan = True )	)
		
		if dtype == 'hist':
			return self._util.scheme_hist( data = amplitude, htype = htype, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$\xi$', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = amplitude, marker = marker, markersize = markersize, is_fit = is_fit, ylabel = ylabel, xlabel = r'$\xi$', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )

	def showAllTAMSD ( self, size = (12,5), is_color = False, xscale = 'linear', yscale = 'linear' ) :

		timemsd = []
		avgmsd = []
		for i in range( len(self.motilitys) ):
			time, average = self.motilitys[i].getAllTAMSD()		
			timemsd.append( time )
			avgmsd.append( average )

		return self._util.scheme_plot_fill( xdata = timemsd, ydata = avgmsd, ystd = [], xlabel = 'Time (min)', ylabel = 'TAMSD', size = size, xscale = xscale, yscale = yscale, is_multi = True, is_color = is_color )

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

	def showTimeAverageMSD ( self, is_multi = False, is_fit = False, window = (0,20), xscale = 'linear', yscale = 'linear' ):

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
				plt.ylabel( r'TAMSD, <TAMSD> $(\mu m^2)$', fontdict = { 'size' : self._font_size } )
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

	def showMSDAndAverageTAMSD( self, is_fit = False, window = (0,20), xscale = 'linear', yscale = 'linear' ):

		def fit_tamsd(t, alpha, org):
			return alpha*t + org

		const_fits = []

		plt.figure( figsize = ( 8, 5 ) )

		for i in range( len(self.motilitys) ):
			time_etamsd, avg_etamsd, std_etamsd = self.motilitys[i].getAverageTAMSD()
			time_emsd, avg_emsd, std_emsd = self.motilitys[i].getMSD()

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

	def showMMEAndAverageTAMME ( self, is_fit = False, window = (0,20), xscale = 'linear', yscale = 'linear' ):

		def fit_tamme(t, alpha, org):
			return alpha*t + org

		const_fits = []

		plt.figure( figsize = ( 8, 5 ) )

		for i in range( len(self.motilitys) ):
			time_etamme, avg_etamme, std_etamme = self.motilitys[i].getAverageTAMME()
			time_emme, avg_emme, std_emme = self.motilitys[i].getMME()	

			plt.plot( time_etamme, avg_etamme, marker = 'o', markeredgecolor = self._colors[i], markersize = 12, linestyle = '', markerfacecolor = (1,1,1,0), lw = 2 )
			plt.plot( time_emme, avg_emme, marker = 's', markeredgecolor = self._colors[i], markersize = 12, linestyle = '', markerfacecolor = (1,1,1,0), lw = 2 )
			
			if is_fit:

				xf_etamme = curve_fit( fit_tamme, numpy.log10(time_etamme[window[0]:window[1]]), numpy.log10( avg_etamme[ window[0]:window[1] ] ) )
				xf_emme = curve_fit( fit_tamme, numpy.log10(time_emme[window[0]:window[1]]), numpy.log10( avg_emme[ window[0]:window[1] ] ) )

				plt.plot( time_etamme[window[0]:window[1]],  numpy.power(time_etamme[window[0]:window[1]], xf_etamme[0][0] )*math.pow(10, xf_etamme[0][1]), linestyle = '-', color = self._colors[i], lw = 2 )
				plt.text( numpy.average(time_etamme[0]), numpy.average(avg_etamme[0]), r'$slope_{<TAMME>}$ ~ '+str( round(xf_etamme[0][0],2) ), color = self._colors[i] )

				plt.plot( time_emme[window[0]:window[1]],  numpy.power(time_emme[window[0]:window[1]], xf_emme[0][0] )*math.pow(10, xf_emme[0][1]), linestyle = ':', color = self._colors[i], lw = 2 )
				plt.text( numpy.average(time_emme[1]), numpy.average(avg_emme[1]), r'$slope_{MME}$ ~ '+str( round(xf_emme[0][0],2) ), color = self._colors[i] )
				
				const_fits.append( ( math.pow(10, xf_etamme[0][1]), xf_etamme[0][0], math.pow(10, xf_emme[0][1]), xf_emme[0][0] ) )

		plt.xscale( xscale )
		plt.yscale( yscale )
		plt.xticks( fontsize = self._font_size )
		plt.yticks( fontsize = self._font_size )
		plt.xlabel( 'time (min)', fontdict = { 'size' : self._font_size } )
		plt.ylabel( r'MME, <TAMME> $(\mu m^2)$', fontdict = { 'size' : self._font_size } )
		plt.grid( linestyle = ':' )

		return const_fits, plt

	# all scaling exponent and diffusion coefficient

	def showDiffusionCoefficientScalingExponentTAMSD ( self, is_set_fit_TAMSD = False, window_TAMSD = (1,6), is_multi = False, is_fit = False, type_fit = 'linear', window = None, markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear' ):

		if is_set_fit_TAMSD :
			for i in range( len(self.motilitys) ):
				self.motilitys[i].setFitTAMSD( window = window_TAMSD )
		
		scaling_fit_values = []
		diffussion_fit_values = []
		for i in range( len(self.motilitys) ):
			scaling_fit_values.append( self.motilitys[i].getScalingExponentFit() )
			diffussion_fit_values.append( self.motilitys[i].getGeneralisedDiffusionCoefficientFit() )

		xsize = ( 6*len(self.motilitys) ,5) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = scaling_fit_values, ydata = diffussion_fit_values, is_fit = is_fit, type_fit = type_fit, window = window, xlabel = r'$ \beta $' , ylabel = r'$ K_{\beta} $', is_multi = is_multi, size = xsize, markersize = markersize, alpha = alpha, xscale = xscale, yscale = yscale )

	def showHistScalingExponent ( self, is_set_fit_TAMSD = False, window = (1,6), dtype = 'hist', is_multi = False, marker = 'o', markersize = 10, is_fit = False, type_fit = 'normal', htype = 'bar', xscale = 'linear', yscale = 'linear', ylabel = 'Count', bins = 10, is_density = False ):
		
		if is_set_fit_TAMSD :
			for i in range( len(self.motilitys) ):
				self.motilitys[i].setFitTAMSD( window = window )

		scaling_fit_values = []
		for i in range( len(self.motilitys) ):
			scaling_fit_values.append( self.motilitys[i].getScalingExponentFit( ) )
		
		if dtype == 'hist' :
			return self._util.scheme_hist( data = scaling_fit_values, htype = htype, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$\beta$', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = scaling_fit_values, marker = marker, markersize = markersize, is_fit = is_fit, ylabel = ylabel, xlabel = r'$\beta$', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
	
	def showInterQuartileScalingExponent ( self, is_set_fit_TAMSD = False, window = (1,6), is_test = False ):

		if is_set_fit_TAMSD :
			for i in range( len(self.motilitys) ):
				self.motilitys[i].setFitTAMSD( window = window )

		scaling_fit_values = []
		for i in range( len(self.motilitys) ):
			scaling_fit_values.append( self.motilitys[i].getScalingExponentFit( ) )

		return self._util.scheme_single_boxplot( data = scaling_fit_values, ylabel = r'$\beta$', is_test = is_test )

	def showHistGeneralizedDiffusionCoefficient ( self, is_set_fit_TAMSD = False, window = (1,6), type_fit = 'powerlaw', dtype = 'hist', is_multi = False, marker = 'o', markersize = 10, is_fit = False, htype = 'bar', xscale = 'linear', yscale = 'linear', ylabel = 'Count', bins = 10, is_density = False ):
		
		if is_set_fit_TAMSD :
			for i in range( len(self.motilitys) ):
				self.motilitys[i].setFitTAMSD( window = window )

		diffusion_fit_values = []
		for i in range( len(self.motilitys) ):
			diffusion_fit_values.append( self.motilitys[i].getGeneralisedDiffusionCoefficientFit( ) )
		
		if dtype == 'hist' :
			return self._util.scheme_hist( data = diffusion_fit_values, htype = htype, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$K_{\beta}$', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = diffusion_fit_values, marker = marker, markersize = markersize, is_fit = is_fit, ylabel = ylabel, xlabel = r'$K_{\beta}$', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )

	def showInterQuartileGeneralizedDiffusionCoefficient ( self, is_set_fit_TAMSD = False, window = (1,6), is_test = False ):

		if is_set_fit_TAMSD :
			for i in range( len(self.motilitys) ):
				self.motilitys[i].setFitTAMSD( window = window )

		diffusion_fit_values = []
		for i in range( len(self.motilitys) ):
			diffusion_fit_values.append( self.motilitys[i].getGeneralisedDiffusionCoefficientFit( ) )

		return self._util.scheme_single_boxplot( data = diffusion_fit_values, ylabel = r'$K_{\beta}$', is_test = is_test )

	def showHistFirstGeneralizedDiffusionCoefficient ( self, is_set_fit_TAMSD = False, window = (1,6), dtype = 'hist', is_multi = False, marker = 'o', markersize = 10, is_fit = False, type_fit = 'normal', htype = 'bar', xscale = 'linear', yscale = 'linear', ylabel = 'Count', bins = 10, is_density = False ):

		if is_set_fit_TAMSD :
			for i in range( len(self.motilitys) ):
				self.motilitys[i].setFitTAMSD( window = window )

		first_diffusion_fit_values = []
		for i in range( len(self.motilitys) ):
			first_diffusion_fit_values.append( self.motilitys[i].getFirstGeneralisedDiffusionCoefficientFit( ) )
		
		if dtype == 'hist' :
			return self._util.scheme_hist( data = first_diffusion_fit_values, htype = htype, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$K_1$', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = first_diffusion_fit_values, marker = marker, markersize = markersize, is_fit = is_fit, ylabel = ylabel, xlabel = r'$K_1$', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )

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

	def showFirstDiffusionCoeffientTimeExperiment ( self, type_time = 'ageing', is_set_fit_TAMSD = False, window_TAMSD = (0,6), is_fit = False, type_fit = 'linear', window = None, is_multi = False, is_legend = False, marker = 'o', markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear' ):
		if is_set_fit_TAMSD :
			for i in range( len(self.motilitys) ):
				self.motilitys[i].setFitTAMSD( window = window_TAMSD )

		start_time = []
		first_diffusion_fit_values = []
		for i in range( len(self.motilitys) ):
			first_diffusion_fit_values.append( self.motilitys[i].getFirstGeneralisedDiffusionCoefficientFit( ) )
			xend = self.motilitys[i].getGlobalEndTime()
			xstart = self.motilitys[i].getGlobalStartTime()
			start_time.append( (xend-xstart)/(xend+xstart) if type_time == 'norm' else xstart/(xend-xstart) )

		xsize = (6*len(self.motilitys),5) if is_multi else (6,5)
		x_label = r'$\frac{t_f-t_i}{t_f+t_i}$' if type_time == 'norm' else r'$t_i/T$'
		return self._util.scheme_scatter( xdata = start_time, ydata = first_diffusion_fit_values, is_fit = is_fit, type_fit = type_fit, window = window, xlabel = x_label, ylabel = r'$K_1$', size = xsize, is_multi = is_multi, is_legend = is_legend, marker = marker, markersize = markersize, alpha = alpha, xscale = xscale , yscale = yscale )

	def showFirstScalingExponentTimeExperiment ( self, type_time = 'ageing', is_set_fit_TAMSD = False, window_TAMSD = (0,6), is_fit = False, type_fit = 'linear', window = None, is_multi = False, is_legend = False, marker = 'o', markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear' ):
		if is_set_fit_TAMSD :
			for i in range( len(self.motilitys) ):
				self.motilitys[i].setFitTAMSD( window = window_TAMSD )

		start_time = []
		first_exponent_fit_values = []
		for i in range( len(self.motilitys) ):
			first_exponent_fit_values.append( self.motilitys[i].getFirstScalingExponentFit( ) )
			xend = self.motilitys[i].getGlobalEndTime()
			xstart = self.motilitys[i].getGlobalStartTime()
			start_time.append( (xend-xstart)/(xend+xstart) if type_time == 'norm' else xstart/(xend-xstart) )

		xsize = (6*len(self.motilitys),5) if is_multi else (6,5)
		x_label = r'$\frac{t_f-t_i}{t_f+t_i}$' if type_time == 'norm' else r'$t_i/T$'
		return self._util.scheme_scatter( xdata = start_time, ydata = first_exponent_fit_values, is_fit = is_fit, type_fit = type_fit, window = window, xlabel = x_label, ylabel = r'${\beta}_1$', size = xsize, is_multi = is_multi, is_legend = is_legend, marker = marker, markersize = markersize, alpha = alpha, xscale = xscale , yscale = yscale )

	def showGeneralizedDiffusionCoefficientTimeExperiment ( self, type_time = 'ageing', is_set_fit_TAMSD = False, window_TAMSD = (0,6), is_fit = False, type_fit = 'linear', window = None, is_multi = False, is_legend = False, marker = 'o', markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear' ):

		if is_set_fit_TAMSD :
			for i in range( len(self.motilitys) ):
				self.motilitys[i].setFitTAMSD( window = window_TAMSD )

		start_time = []
		diffusion_fit_values = []
		for i in range( len(self.motilitys) ):
			diffusion_fit_values.append( self.motilitys[i].getGeneralisedDiffusionCoefficientFit( ) )
			xend = self.motilitys[i].getGlobalEndTime()
			xstart = self.motilitys[i].getGlobalStartTime()
			start_time.append( (xend-xstart)/(xend+xstart) if type_time == 'norm' else xstart/(xend-xstart) )

		xsize = (6*len(self.motilitys),5) if is_multi else (6,5)
		x_label = r'$\frac{t_f-t_i}{t_f+t_i}$' if type_time == 'norm' else r'$t_i/T$'
		return self._util.scheme_scatter( xdata = start_time, ydata = diffusion_fit_values, is_fit = is_fit, type_fit = type_fit, window = window, xlabel = x_label, ylabel = r'$K_{\beta}$', size = xsize, is_multi = is_multi, is_legend = is_legend, marker = marker, markersize = markersize, alpha = alpha, xscale = xscale , yscale = yscale )

	def showScalingExponentTimeExperiment ( self, type_time = 'ageing', is_set_fit_TAMSD = False, window_TAMSD = (0,6), is_fit = False, type_fit = 'linear', window = None, is_multi = False, is_legend = False, marker = 'o', markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear' ):

		if is_set_fit_TAMSD :
			for i in range( len(self.motilitys) ):
				self.motilitys[i].setFitTAMSD( window = window_TAMSD )

		start_time = []
		exponent_fit_values = []
		for i in range( len(self.motilitys) ):
			exponent_fit_values.append( self.motilitys[i].getScalingExponentFit( ) )
			xend = self.motilitys[i].getGlobalEndTime()
			xstart = self.motilitys[i].getGlobalStartTime()
			start_time.append( (xend-xstart)/(xend+xstart) if type_time == 'norm' else xstart/(xend-xstart) )

		xsize = (6*len(self.motilitys),5) if is_multi else (6,5)
		x_label = r'$\frac{t_f-t_i}{t_f+t_i}$' if type_time == 'norm' else r'$t_i/T$'
		return self._util.scheme_scatter( xdata = start_time, ydata = exponent_fit_values, is_fit = is_fit, type_fit = type_fit, window = window, xlabel = x_label, ylabel = r'$\beta$', size = xsize, is_multi = is_multi, is_legend = is_legend, marker = marker, markersize = markersize, alpha = alpha, xscale = xscale , yscale = yscale )

	# persistence with ageing

	def showFractalTimeExperiment ( self, type_time = 'ageing', is_multi = False, is_fit = False, type_fit = 'linear', window = None, markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear' ) :

		start_time = []
		fractal = []
		for i in range( len(self.motilitys) ):
			fractal.append( self.motilitys[i].getGlobalFractalDimension() )
			xend = self.motilitys[i].getGlobalEndTime()
			xstart = self.motilitys[i].getGlobalStartTime()
			start_time.append( (xend-xstart)/(xend+xstart) if type_time == 'norm' else xstart/(xend-xstart) )

		xsize = (6*len(self.motilitys),5) if is_multi else (6,5)
		x_label = r'$\frac{t_f-t_i}{t_f+t_i}$' if type_time == 'norm' else r'$t_i/T$'
		return self._util.scheme_scatter( xdata = start_time, ydata = fractal, is_fit = is_fit, type_fit = type_fit, window = window, xlabel = x_label , ylabel = r'$d_f$', is_multi = is_multi, size = xsize, markersize = markersize, alpha = alpha, xscale = xscale, yscale = yscale )

	def showConfinementRatioTimeExperiment ( self, type_time = 'ageing', is_multi = False, is_fit = False, type_fit = 'linear', window = None, markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear' ):

		start_time = []
		ratio_confinement = []
		for i in range( len(self.motilitys) ):
			ratio_confinement.append( self.motilitys[i].getGlobalPersistenceRatio() )
			xend = self.motilitys[i].getGlobalEndTime()
			xstart = self.motilitys[i].getGlobalStartTime()
			start_time.append( (xend-xstart)/(xend+xstart) if type_time == 'norm' else xstart/(xend-xstart) )

		xsize = (6*len(self.motilitys),5) if is_multi else (6,5)
		x_label = r'$\frac{t_f-t_i}{t_f+t_i}$' if type_time == 'norm' else r'$t_i/T$'
		return self._util.scheme_scatter( xdata = start_time, ydata = ratio_confinement, is_fit = is_fit, type_fit = type_fit, window = window, xlabel = x_label , ylabel = 'Ratio Confinement', is_multi = is_multi, size = xsize, markersize = markersize, alpha = alpha, xscale = xscale, yscale = yscale )

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

	def showHistAmplitudeTAMSDWithAgeing ( self, dtype = 'hist', is_multi = False, marker = 'o', markersize = 10, is_fit = False, type_fit = 'normal', htype = 'bar', xscale = 'linear', yscale = 'linear', ylabel = 'Count', bins = 10, is_density = False ):

		amplitude = []
		for i in range( len(self.motilitys) ):
			amplitude.append( self.motilitys[i].getAmplitudeTAMSDWithAging( is_flat_not_nan = True )	)
		
		if dtype == 'hist':
			return self._util.scheme_hist( data = amplitude, htype = htype, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$\xi$', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = amplitude, marker = marker, markersize = markersize, is_fit = is_fit, ylabel = ylabel, xlabel = r'$\xi$', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )

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
			mixing.append( numpy.real(xmixing)[ xend_index-xstart_index, range( xmixing.shape[1] ) ] )
			ergodicity.append( numpy.real(xergo)[ xend_index-xstart_index, range( xergo.shape[1] ) ] )

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
			mixing.append( numpy.real(ymixing)[ yend_index-ystart_index, range( ymixing.shape[1] ) ] )
			ergodicity.append( numpy.real(yergo)[ yend_index-ystart_index, range( yergo.shape[1] ) ] )

		xsize = (6*len(self.motilitys), 5) if is_multi else (6, 5)
		yinfo_mix, ypltmix = self._util.scheme_scatter( xdata = exponent, ydata = mixing, is_multi = is_multi, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit , window = window, alpha = alpha, marker = marker, markersize = markersize, size = xsize, xlabel = r'$\beta$', ylabel = r'$Re\left( \hat E(n) \right)$', xscale = xscale, yscale = yscale )
		yinfo_ergo, ypltergo = self._util.scheme_scatter( xdata = exponent, ydata = ergodicity, is_multi = is_multi, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit , window = window, alpha = alpha, marker = marker, markersize = markersize, size = xsize, xlabel = r'$\beta$', ylabel = r'$Re\left( \sum_{k=0}^{n-1} \hat E(k)/n \right)$', xscale = xscale, yscale = yscale )

		return yinfo_mix, ypltmix, yinfo_ergo, ypltergo

	# all persistence

	def showDiffusionCoefficientConfinementRatio ( self, is_set_fit_TAMSD = False, window_TAMSD = (1,6), is_fit = False,is_fit_all = False, type_fit = 'linear', window = None, is_multi = False, markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear' ):
		
		if is_set_fit_TAMSD :
			for i in range( len(self.motilitys) ):
				self.motilitys[i].setFitTAMSD( window = window_TAMSD )
		
		ratio_confinement = []
		diffussion_fit_values = []
		for i in range( len(self.motilitys) ):
			ratio_confinement.append( self.motilitys[i].getGlobalPersistenceRatio() )
			diffussion_fit_values.append( self.motilitys[i].getGeneralisedDiffusionCoefficientFit() )

		xsize = ( 6*len(ratio_confinement) ,5) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = ratio_confinement, ydata = diffussion_fit_values, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit, window = window, xlabel = 'Ratio Confinement' , ylabel = r'$ K_{\beta} $', is_multi = is_multi, size = xsize, markersize = markersize, alpha = alpha, xscale = xscale, yscale = yscale )

	def showScalingExponentConfinementRatio ( self, is_set_fit_TAMSD = False, window_TAMSD = (1,6), is_fit = False, is_fit_all = False, type_fit = 'linear', window = None, is_multi = False, markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear' ):
		
		if is_set_fit_TAMSD :
			for i in range( len(self.motilitys) ):
				self.motilitys[i].setFitTAMSD( window = window_TAMSD )
		
		ratio_confinement = []
		exponent_fit_values = []
		for i in range( len(self.motilitys) ):
			ratio_confinement.append( self.motilitys[i].getGlobalPersistenceRatio() )
			exponent_fit_values.append( self.motilitys[i].getScalingExponentFit() )

		xsize = ( 6*len(ratio_confinement) ,5) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = ratio_confinement, ydata = exponent_fit_values, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit, window = window, xlabel = 'Ratio Confinement' , ylabel = r'$\beta$', is_multi = is_multi, size = xsize, markersize = markersize, alpha = alpha, xscale = xscale, yscale = yscale )
	
	def showDiffusionCoefficientFractalDimension ( self, is_set_fit_TAMSD = False, window_TAMSD = (1,6), is_fit = False, is_fit_all = False, type_fit = 'linear', window = None, is_multi = False, markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear' ):

		if is_set_fit_TAMSD :
			for i in range( len(self.motilitys) ):
				self.motilitys[i].setFitTAMSD( window = window_TAMSD )
		
		fractal = []
		diffussion_fit_values = []
		for i in range( len(self.motilitys) ):
			fractal.append( self.motilitys[i].getGlobalFractalDimension() )
			diffussion_fit_values.append( self.motilitys[i].getGeneralisedDiffusionCoefficientFit() )

		xsize = ( 6*len(diffussion_fit_values) ,5) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = fractal, ydata = diffussion_fit_values, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit, window = window, xlabel = r'$d_f$' , ylabel = r'$ K_{\beta} $', is_multi = is_multi, size = xsize, markersize = markersize, alpha = alpha, xscale = xscale, yscale = yscale )

	def showScalingExponentFractalDimension ( self, is_set_fit_TAMSD = False, window_TAMSD = (1,6), is_fit = False, is_fit_all = False, type_fit = 'linear', window = None, is_multi = False, markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear' ) :

		if is_set_fit_TAMSD :
			for i in range( len(self.motilitys) ):
				self.motilitys[i].setFitTAMSD( window = window_TAMSD )
		
		exponent_fit_values = []
		fractal = []
		for i in range( len(self.motilitys) ):
			exponent_fit_values.append( self.motilitys[i].getScalingExponentFit() )
			fractal.append( self.motilitys[i].getGlobalFractalDimension() )

		xsize = ( 6*len(exponent_fit_values) ,5) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = fractal, ydata = exponent_fit_values, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit, window = window, xlabel = r'$d_f$' , ylabel = r'$\beta$', is_multi = is_multi, size = xsize, markersize = markersize, alpha = alpha, xscale = xscale, yscale = yscale )

	def showFractalDimensionConfinementRatio ( self, is_multi = False, is_fit = False, is_fit_all = False, type_fit = 'linear', window = None, markersize = 100, alpha = 0.5, xscale = 'linear', yscale = 'linear' ) :

		ratio_confinement = []
		fractal = []
		for i in range( len(self.motilitys) ):
			ratio_confinement.append( self.motilitys[i].getGlobalPersistenceRatio() )
			fractal.append( self.motilitys[i].getGlobalFractalDimension() )

		xsize = ( 6*len(ratio_confinement) ,5) if is_multi else (6,5)
		return self._util.scheme_scatter( xdata = ratio_confinement, ydata = fractal, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit, window = window, xlabel = 'Ratio Confinement' , ylabel = r'$d_f$', is_multi = is_multi, size = xsize, markersize = markersize, alpha = alpha, xscale = xscale, yscale = yscale )

	def showHistFractalDimension ( self, dtype = 'hist', is_multi = False, marker = 'o', markersize = 10, is_fit = False, type_fit = 'normal', htype = 'bar', xscale = 'linear', yscale = 'linear', ylabel = 'Count', bins = 10, is_density = False ):
		
		fractal = []
		for i in range( len(self.motilitys) ):
			fractal.append( self.motilitys[i].getGlobalFractalDimension( ) )
		
		if dtype == 'hist' :
			return self._util.scheme_hist( data = fractal, htype = htype, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$d_f$', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = fractal, marker = marker, markersize = markersize, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$d_f$', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )

	def showHistPersistenceRatio ( self, dtype = 'hist', is_multi = False, marker = 'o', markersize = 10, is_fit = False, type_fit = 'normal', htype = 'bar', xscale = 'linear', yscale = 'linear', ylabel = 'Count', bins = 10, is_density = False ):

		persistenceratio = []
		for i in range( len(self.motilitys) ):
			persistenceratio.append( self.motilitys[i].getGlobalPersistenceRatio() )

		if dtype == 'hist' :
			return self._util.scheme_hist( data = persistenceratio, htype = htype, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = 'Ratio Confinement', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = persistenceratio, marker = marker, markersize = markersize, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = 'Ratio Confinement', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )

	def showHistGyrationRadius ( self, dtype = 'hist', is_multi = False, marker = 'o', markersize = 10, is_fit = False, type_fit = 'normal', htype = 'bar', xscale = 'linear', yscale = 'linear', ylabel = 'Count', bins = 10, is_density = False ):

		gyration_radius = []
		for i in range( len(self.motilitys) ):
			gyration_radius.append( self.motilitys[i].getGyrationRadius() )
		
		if dtype == 'hist':
			return self._util.scheme_hist( data = gyration_radius, htype = htype, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$R_G^2 \; (\mu m ^ 2)$', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = gyration_radius, marker = marker, markersize = markersize, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$R_G^2 \; (\mu m ^ 2)$', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )

	def showHistGyrationAsymmetryRatio_a2 ( self, dtype = 'hist', is_multi = False, marker = 'o', markersize = 10, is_fit = False, type_fit = 'normal', htype = 'bar', xscale = 'linear', yscale = 'linear', ylabel = 'Count', bins = 10, is_density = False ):

		asymmetryratio_a2 = []
		for i in range( len(self.motilitys) ):
			asymmetryratio_a2.append( self.motilitys[i].getGyrationAsymmetryRatio_a2( ) )
		
		if dtype == 'hist' :
			return self._util.scheme_hist( data = asymmetryratio_a2, htype = htype, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$a_2$', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = asymmetryratio_a2, marker = marker, markersize = markersize, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$a_2$', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )		

	def showHistGyrationAsymmetryRatio_A2 ( self, dtype = 'hist', is_multi = False, marker = 'o', markersize = 10, is_fit = False, type_fit = 'normal', htype = 'bar', xscale = 'linear', yscale = 'linear', ylabel = 'Count', bins = 10, is_density = False ):

		asymmetryratio_A2 = []
		for i in range( len(self.motilitys) ):
			asymmetryratio_A2.append( self.motilitys[i].getGyrationAsymmetryRatio_A2( ) )
		
		if dtype == 'hist' :
			return self._util.scheme_hist( data = asymmetryratio_A2, htype = htype, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$A_2$', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = asymmetryratio_A2, marker = marker, markersize = markersize, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$A_2$', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )

	def showHistGyrationAsymmetry_A ( self, dtype = 'hist', is_multi = False, marker = 'o', markersize = 10, is_fit = False, type_fit = 'normal', htype = 'bar', xscale = 'linear', yscale = 'linear', ylabel = 'Count', bins = 10, is_density = False ):

		asymmetry_A = []
		for i in range( len(self.motilitys) ):
			asymmetry_A.append( self.motilitys[i].getGyrationAsummetry_A( ) )
		
		if dtype == 'hist' :
			return self._util.scheme_hist( data = asymmetry_A, htype = htype, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = 'A', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = asymmetry_A, marker = marker, markersize = markersize, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = 'A', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )

	def showHistConvexHullPerimeter ( self, dtype = 'hist', is_multi = False, marker = 'o', markersize = 10, is_fit = False, type_fit = 'normal', htype = 'bar', xscale = 'linear', yscale = 'linear', ylabel = 'Count', bins = 10, is_density = False ):

		convex_hull_perimeter = []
		for i in range( len(self.motilitys) ):
			convex_hull_perimeter.append( self.motilitys[i].getGlobalHullPerimeter( ) )
		
		if dtype == 'hist' :
			return self._util.scheme_hist( data = convex_hull_perimeter, htype = htype, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$P \; (\mu m)$', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = convex_hull_perimeter, marker = marker, markersize = markersize, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$P \; (\mu m)$', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )

	def showHistConvexHullArea ( self, dtype = 'hist', is_multi = False, marker = 'o', markersize = 10, is_fit = False, type_fit = 'normal', htype = 'bar', xscale = 'linear', yscale = 'linear', ylabel = 'Count', bins = 10, is_density = False ):

		convex_hull_area = []
		for i in range( len(self.motilitys) ):
			convex_hull_area.append( self.motilitys[i].getGlobalHullArea( ) )
		
		if dtype == 'hist' :
			return self._util.scheme_hist( data = convex_hull_area, htype = htype, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$A \; (\mu m^2)$', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = convex_hull_area, marker = marker, markersize = markersize, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = r'$A \; (\mu m^2)$', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
	
	def showHistConvexHullAcircularity ( self, dtype = 'hist', is_multi = False, marker = 'o', markersize = 10, is_fit = False, type_fit = 'normal', htype = 'bar', xscale = 'linear', yscale = 'linear', ylabel = 'Count', bins = 10, is_density = False ):

		convex_hull_acir = []
		for i in range( len(self.motilitys) ):
			convex_hull_acir.append( self.motilitys[i].getGlobalAcircularity( ) )
		
		if dtype == 'hist' :
			return self._util.scheme_hist( data = convex_hull_acir, htype = htype, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = 'a', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )
		elif dtype == 'scatter':
			return self._util.scheme_scatter_hist( data = convex_hull_acir, marker = marker, markersize = markersize, is_fit = is_fit, type_fit = type_fit, ylabel = ylabel, xlabel = 'a', density = is_density, is_legend = True, is_multi = is_multi, bins = bins, xscale = xscale, yscale = yscale )

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

		return self._util.scheme_plot_fill( xdata = timepersistence, ydata = avgpersistence, ystd = stdpersistence, xlabel = 'Time (min)', ylabel = 'Persistence', is_legend = True )
	
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

	def showMomentScalingExponent ( self, exponents = [2], window_moment = (0,10), is_fit = False, window = None, type_fit = 'linear', is_multi = False, is_color = False, is_legend = False, xscale = 'linear', yscale = 'linear'  ):

		def fit_tamsd(t, alpha, org):
			return alpha*t + org

		xdata = []
		ydata = []
		for i in range( len(self.motilitys) ):
			time, moments = self.motilitys[i].getMomentsMSD( scaling_exponent = exponents )	
			
			scaling_exponent = []
			diffusion_coefficient = []
			for j, xmoments in enumerate(moments):
				xrest = curve_fit( fit_tamsd, numpy.log10(time[window_moment[0]:window_moment[1]]), numpy.log10( xmoments[ window_moment[0]:window_moment[1] ] ) )
				scaling_exponent.append( xrest[0][0] )
				diffusion_coefficient.append( math.pow(10, xrest[0][1]) )
			
			xdata.append( numpy.array(exponents) )
			ydata.append( numpy.array(scaling_exponent) )

		xsize = (6*len(self.motilitys), 5) if is_multi else (6, 5)
		return self._util.scheme_scatter( xdata = xdata, ydata = ydata, xlabel = 'q', ylabel = r'$\zeta(q)$', size = xsize, alpha = 0.5, is_fit = is_fit, window = window, type_fit = type_fit, is_multi = is_multi, is_color = is_color, is_legend = is_legend, xscale = xscale, yscale = yscale )

	# gaussianity parameter

	def showGaussianityParameter ( self, is_multi= False, is_color = False, is_legend = False, xscale = 'linear', yscale = 'linear' ):
		
		time = []
		gauss = []
				
		for i in range( len(self.motilitys) ):
			xtime, xgass = self.motilitys[i].getGaussianityParameter()	
			time.append( xtime )
			gauss.append( xgass )
		
		xsize = (6*len(self.motilitys), 5) if is_multi else (6, 5)
		xinfo_fit, xplt = self._util.scheme_scatter( xdata = time, ydata = gauss, xlabel = r'$\Delta \; (min)$', ylabel = 'G', size = xsize, alpha = 0.5, is_multi = is_multi, is_color = is_color, is_legend = is_legend, xscale = xscale, yscale = yscale )
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
			xtime, xpc, xcell = self.motilitys[i].getPackingCoefficient( length_window = length_window )
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

	# persistence time 

	def showEndToEndDistanceVsTimePersistence ( self, is_multi= False, is_fit = False, is_fit_all = False, type_fit = 'linear', window = None, is_color = False, is_legend = False, xscale = 'linear', yscale = 'linear' ) :

		time = []
		endtoend = []
				
		for i in range( len(self.motilitys) ):
			xdata = self.motilitys[i].getDataPersistenceTime()
			time.append( xdata[:,0] )
			endtoend.append( xdata[:,1] )
		
		xsize = (6*len(self.motilitys), 5) if is_multi else (6, 5)
		return self._util.scheme_scatter( xdata = time, ydata = endtoend, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit, window = window, xlabel = r'$ t_p \; (min)$', ylabel = r'$\mathcal{L}_P^{\;end\; to \; end} \; (\mu m) $', size = xsize, alpha = 0.5, is_multi = is_multi, is_color = is_color, is_legend = is_legend, xscale = xscale, yscale = yscale )
	
	def showNetDistanceVsTimePersistence ( self, is_multi= False, is_fit = False, is_fit_all = False, type_fit = 'linear', window = None, is_color = False, is_legend = False, xscale = 'linear', yscale = 'linear' ) :

		time = []
		distance = []
				
		for i in range( len(self.motilitys) ):
			xdata = self.motilitys[i].getDataPersistenceTime()
			time.append( xdata[:,0] )
			distance.append( xdata[:,2] )
		
		xsize = (6*len(self.motilitys), 5) if is_multi else (6, 5)
		return self._util.scheme_scatter( xdata = time, ydata = distance, is_fit = is_fit, is_fit_all = is_fit_all, type_fit = type_fit, window = window, xlabel = r'$ t_p \; (min)$', ylabel = r'$\mathcal{L}_P^{\;net \; distance} \; (\mu m) $', size = xsize, alpha = 0.5, is_multi = is_multi, is_color = is_color, is_legend = is_legend, xscale = xscale, yscale = yscale )

	# global FMI

	def showGlobalFMIx ( self, is_test = False ) :

		fmi_x = []
		for i in range( len(self.motilitys) ):
			fmi_x.append( self.motilitys[i].getGlobalXFMI() )

		return self._util.scheme_single_boxplot( data = fmi_x, ylabel = r'$FMI_x$', is_test = is_test )

	def showGlobalFMIy ( self, is_test = False ):
		
		fmi_y = []
		for i in range( len(self.motilitys) ):
			fmi_y.append( self.motilitys[i].getGlobalYFMI() )

		return self._util.scheme_single_boxplot( data = fmi_y, ylabel = r'$FMI_y$', is_test = is_test )

	def showGlobalFMI ( self, is_test = False, showfliers = True ):

		fmi_x = []
		fmi_y = []
		for i in range( len(self.motilitys) ):
			fmi_x.append( self.motilitys[i].getGlobalXFMI() )
			fmi_y.append( self.motilitys[i].getGlobalYFMI() )

		return self._util.scheme_multiple_boxplot( data = [ fmi_x, fmi_y ], xlabels = self._names , ylabel = r'$FMI$', spancap = 3, color_box = [ 'darkorange','bisque' ], is_legend = True, label_legend = [r'$FMI_x$',r'$FMI_y$'], showfliers = showfliers, is_test = is_test )		

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

	def showGlobalDirectionalityRatio ( self, is_test = False, showfliers = True ):

		directratio = []
		for i in range( len(self.motilitys) ):
			directratio.append( self.motilitys[i].getGlobalDirectionalityRatio( is_not_nan = True ) )

		return self._util.scheme_single_boxplot( data = directratio, ylabel = 'Directionality ratio', is_test = is_test, showfliers = showfliers )

	def showGlobalDisplacementRatio ( self, is_test = False ):

		displacratio = []
		for i in range( len(self.motilitys) ):
			displacratio.append( self.motilitys[i].getGlobalDisplacementRatio() )

		return self._util.scheme_single_boxplot( data = directratio, ylabel = 'Displacement ratio', is_test = is_test )

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

	def showGlobalXYMeanStraightLineSpeed ( self, is_test = False, showfliers = True ):

		xspeed = []
		yspeed = []
		for i in range( len(self.motilitys) ):
			xspeed.append( self.motilitys[i].getGlobalXMeanStraightLineSpeed() )
			yspeed.append( self.motilitys[i].getGlobalYMeanStraightLineSpeed() )

		return self._util.scheme_multiple_boxplot( data = [ xspeed, yspeed ], xlabels = self._names , ylabel = 'Speed ($\mu m/min$)', is_legend = True, color_box = [ 'maroon', 'red' ], label_legend = [ r'$\nu_x$', r'$\nu_y$' ], showfliers = showfliers, spancap = 3, is_test = is_test )

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

		return plt

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

	# Global first orientation body cell
	def showGlobalFirstOrientationCellBody ( self, size = (12,4), bin_size = 20 ):

		angle = []
		for i in range( len(self.motilitys) ):
			angle.append( self.motilitys[i].getGlobalFirstOrientationCellBody() )

		return self._util.scheme_polar_histogram( data = angle, size = size, bin_size = bin_size )


