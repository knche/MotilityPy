import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy
import math
from scipy import stats
from scipy.optimize import curve_fit
from scipy.stats import gamma, kurtosis, skew, pearsonr, laplace, norm, cauchy, chi, gengamma, expon, pareto, powerlaw, loguniform, lognorm, chi2, pearson3, betaprime, genpareto, nct as ncstudent, t as student, studentized_range

class util :

	def __init__ ( self, colors = [], names = [], font_size = 16, interval_axis_plot = 6, test_threshold_value = 0.01 ):

		self._colors = colors
		self._names = names
		self._font_size = font_size
		self._interval_axis_plot = interval_axis_plot
		self._test_threshold_value = test_threshold_value

	def setColors ( self, colors = [] ) :
		self._colors = colors
	def setNames ( self, names = [] ) :
		self._names = names
	def setIntervalAxisPlot ( self, interval_axis_plot = 6 ):
		self._interval_axis_plot = interval_axis_plot
	def setFontSize ( self, font_size = 16 ):
		self._font_size = font_size
	def setTestThresholdValue ( self, test_threshold_value = 0.01 ) :
		self._test_threshold_value = test_threshold_value

	def scheme_polar_histogram ( self, data = [], size = (12,4), bin_size = 20 ):

		plt.figure( figsize = size )

		for i in range( len(data) ):
			plt.subplot(1, len(data), i+1 , projection = 'polar')

			plt.title(self._names[i]+' \n', fontdict = { 'fontsize' : 16 })

			a , b = numpy.histogram( data[i] , bins=numpy.arange(0, 360+bin_size, bin_size))
			centers = numpy.deg2rad(numpy.ediff1d(b)//2 + b[:-1])

			plt.bar(centers, a, width = numpy.deg2rad(bin_size), bottom=0.0, color = self._colors[i] , edgecolor='k')

		plt.tight_layout()

		return plt

	def scheme_scatter ( self, is_multi = False, is_fit = False, is_fit_all = False, type_fit = 'linear', window = None, is_axis_equal = False, alpha = 0.2, marker = 'o', markersize = 100, size = (12,5), is_color = False, xdata = [], ydata = [], xlabel = '', ylabel = '', is_legend = False, xscale = 'linear', yscale = 'linear' ) :

		plt.figure( figsize = size )

		info_fit = []
		multi_data = []
		if not is_multi and is_fit and is_fit_all and len(xdata)>0:
			multi_data = numpy.stack( ( xdata[0], ydata[0] ), axis = -1 )

		for i in range( len(xdata) ):
			if is_multi :
				plt.subplot(1, len(xdata), i+1)

			xcolor = self._colors[i]
			if is_color :
				xcolor = None

			if not is_multi and is_fit and is_fit_all and i > 0:
				multi_data = numpy.concatenate( ( multi_data, numpy.stack( ( xdata[i], ydata[i] ), axis = -1 ) ), axis = 0 )

			plt.scatter( xdata[i], ydata[i], c = xcolor, label = self._names[i], alpha = alpha, s = markersize, marker = marker )

			if is_fit and not is_fit_all:
				xfit = xdata[i][window[0]:window[1]] if window else xdata[i]
				yfit = ydata[i][window[0]:window[1]] if window else ydata[i]
				i_nonnan = numpy.where( ~numpy.isnan(yfit) )[0]
				xfit = xfit[i_nonnan]
				yfit = yfit[i_nonnan]
				i_nonnan = numpy.where( ~numpy.isnan(xfit) )[0]
				xfit = xfit[i_nonnan]
				yfit = yfit[i_nonnan]
				i_noninf = numpy.where( ~numpy.isinf(yfit) )[0]
				xfit = xfit[i_noninf]
				yfit = yfit[i_noninf]
				i_noninf = numpy.where( ~numpy.isinf(xfit) )[0]
				xfit = xfit[i_noninf]
				yfit = yfit[i_noninf]
				if type_fit == 'linear' :
					def fit_linear ( x, m, c ):
						return x*m + c
					xrest = curve_fit( fit_linear, xfit, yfit )
					plt.plot( xfit, xrest[0][0]*xfit + xrest[0][1], color = self._colors[i], linestyle ='-', lw = 2 )

					info_fit.append( { 'slope' : xrest[0][0], 'intersection' : xrest[0][1], 'pearson': pearsonr( xfit, yfit )} )

				elif type_fit == 'powerlaw':
					def fit_powerlaw ( x, m, c ):
						return x*m + c
					xrest = curve_fit( fit_powerlaw, numpy.log10(xfit), numpy.log10(yfit) )
					plt.plot( xfit,  numpy.power(xfit, xrest[0][0] )*math.pow(10, xrest[0][1]), linestyle = '-', color = self._colors[i], lw = 2 )
					
					info_fit.append( { 'slope' : xrest[0][0], 'const' : math.pow(10, xrest[0][1]), 'pearson': pearsonr( numpy.log10(xfit), numpy.log10(yfit) )} )
				elif type_fit == 'exponential':
					def fit_exponential ( x, m, c ):
						return x*m*math.log10(math.e) + c
					yfit = numpy.log10(yfit)
					i_noninf = numpy.where( ~numpy.isinf(yfit) )[0]
					xfit = xfit[i_noninf]
					yfit = yfit[i_noninf]
					i_nonnan = numpy.where( ~numpy.isnan(yfit) )[0]
					xfit = xfit[i_nonnan]
					yfit = yfit[i_nonnan]
					xrest = curve_fit( fit_exponential, xfit, yfit )
					plt.plot( xfit,  numpy.exp( xrest[0][0]*xfit)*math.pow(10, xrest[0][1]), linestyle = '-', color = 'black', lw = 2 )
					
					info_fit.append( { 'exponent' : xrest[0][0], 'const' : math.pow(10, xrest[0][1]), 'pearson': pearsonr( xfit, yfit )} )
			
			if is_multi :
				if is_axis_equal :
					plt.axis('equal')
				plt.xticks( fontsize = self._font_size )
				plt.yticks( fontsize = self._font_size )
				plt.xscale( xscale )
				plt.yscale( yscale )
				plt.xlabel( xlabel, fontdict = { 'size' : self._font_size } )
				plt.ylabel( ylabel, fontdict = { 'size' : self._font_size } )
				plt.grid( linestyle = ':' )

		if not is_multi :
			if is_fit and is_fit_all :
				xfit = multi_data[:,0]
				yfit = multi_data[:,1]
				i_nonnan = numpy.where( ~numpy.isnan(yfit) )[0]
				xfit = xfit[i_nonnan]
				yfit = yfit[i_nonnan]
				i_nonnan = numpy.where( ~numpy.isnan(xfit) )[0]
				xfit = xfit[i_nonnan]
				yfit = yfit[i_nonnan]
				i_noninf = numpy.where( ~numpy.isinf(yfit) )[0]
				xfit = xfit[i_noninf]
				yfit = yfit[i_noninf]
				i_noninf = numpy.where( ~numpy.isinf(xfit) )[0]
				xfit = xfit[i_noninf]
				yfit = yfit[i_noninf]

				xedge = numpy.linspace( numpy.min(xfit), numpy.max(xfit), 100 )
				if type_fit == 'linear' :
					def fit_linear ( x, m, c ):
						return x*m + c
					xrest = curve_fit( fit_linear, xfit, yfit )
					plt.plot( xedge, xrest[0][0]*xedge + xrest[0][1], color = 'black', linestyle ='-', lw = 2 )

					info_fit.append( { 'slope' : xrest[0][0], 'intersection' : xrest[0][1], 'pearson': pearsonr( xfit, yfit )} )

				elif type_fit == 'powerlaw':
					def fit_powerlaw ( x, m, c ):
						return x*m + c
					xrest = curve_fit( fit_powerlaw, numpy.log10(xfit), numpy.log10(yfit) )
					plt.plot( xedge,  numpy.power(xedge, xrest[0][0] )*math.pow(10, xrest[0][1]), linestyle = '-', color = 'black', lw = 2 )
					
					info_fit.append( { 'slope' : xrest[0][0], 'const' : math.pow(10, xrest[0][1]), 'pearson': pearsonr( numpy.log10(xfit), numpy.log10(yfit) )} )
				elif type_fit == 'exponential':
					def fit_exponential ( x, m, c ):
						return x*m*math.log10(math.e) + c
					yfit = numpy.log10(yfit)
					i_noninf = numpy.where( ~numpy.isinf(yfit) )[0]
					xfit = xfit[i_noninf]
					yfit = yfit[i_noninf]
					i_nonnan = numpy.where( ~numpy.isnan(yfit) )[0]
					xfit = xfit[i_nonnan]
					yfit = yfit[i_nonnan]
					xrest = curve_fit( fit_exponential, xfit, yfit )
					plt.plot( xedge,  numpy.exp( xrest[0][0]*xedge)*math.pow(10, xrest[0][1]), linestyle = '-', color = 'black', lw = 2 )
					
					info_fit.append( { 'exponent' : xrest[0][0], 'const' : math.pow(10, xrest[0][1]), 'pearson': pearsonr( xfit, yfit )} )

			if is_axis_equal :
				plt.axis('equal')
			plt.xticks( fontsize = self._font_size )
			plt.yticks( fontsize = self._font_size )
			plt.xscale( xscale )
			plt.yscale( yscale )
			plt.xlabel( xlabel, fontdict = { 'size' : self._font_size } )
			plt.ylabel( ylabel, fontdict = { 'size' : self._font_size } )
			plt.grid( linestyle = ':' )
			if is_legend :
				plt.legend( frameon = False )
		else:
			plt.tight_layout()

		return info_fit, plt

	def scheme_scatter_hist ( self, data = [], bins = 10, is_fit = False, type_fit = 'normal', marker = 'o', markersize = 100, xscale = 'linear', yscale = 'linear', density = False, xlabel = '', ylabel = '', is_multi = False, is_legend = False ):

		if not is_multi:
			plt.figure( figsize = ( 6 , 5) )
		else:
			plt.figure( figsize = ( 6*len(data) , 5) )

		info = []
		for i in range( len(data) ):
			if is_multi :
				plt.subplot(1, len(data), i+1)


			weights, bins_edges = numpy.histogram( data[i], bins = bins, density = density ) 
			positions = bins_edges[0:-1:] + (bins_edges[1::] - bins_edges[0:-1:])/2

			plt.plot( positions, weights, linestyle = '', marker = marker, markersize = markersize, markerfacecolor = self._colors[i], color = self._colors[i], label = self._names[i] )

			mu_data = numpy.nanmean( data[i] )
			sigma_data = math.sqrt( numpy.nanvar( data[i] ) )
			xmedian = numpy.nanmedian( data[i] )
			xindex_mode = numpy.argmax( weights )
			xmode = bins_edges[xindex_mode] + (bins_edges[xindex_mode+1] - bins_edges[xindex_mode])/2
			xkurt = kurtosis( data[i] )
			xske = skew( data[i] )
			xinfo = { 'mean' : mu_data, 'median' : xmedian, 'mode' : xmode , 'standar_desviation' : sigma_data, 'kurtosis' : xkurt, 'skewness' : xske }

			if is_fit:
				xtype_fit = [type_fit] if type(type_fit) is str else type_fit
				xls = ['dotted','dashed','dashdot','solid','loosely dotted','loosely dashed']
				for j, xtype in enumerate(xtype_fit):
					if density and xtype == 'normal' :
						xrest = norm.fit( data[i] )
						xedges = numpy.linspace( bins_edges[0], bins_edges[-1], 100 )
						plt.plot( xedges, norm.pdf( xedges, xrest[0], xrest[1] ), linestyle = xls[j], color = xcolor, lw = 4 )
						xinfo['fit_norm'] = {'loc':xrest[0],'scale':xrest[1]}
					elif xtype == 'powerlaw':
						def fit_powerlaw ( x, m, c ):
							return x*m + c
						xfit = numpy.log10(positions)
						yfit = numpy.log10(weights)
						i_noninf = numpy.where( ~numpy.isinf(yfit) )[0]
						xfit = xfit[i_noninf]
						yfit = yfit[i_noninf]
						xrest = curve_fit( fit_powerlaw, xfit, yfit )
						xedges = numpy.linspace( positions[0], positions[-1], 100 )
						plt.plot( xedges, numpy.power(xedges, xrest[0][0] )*math.pow(10, xrest[0][1]), linestyle = xls[j], color = self._colors[i], lw = 2 )
						xinfo['fit_powerlaw'] = {'slope':xrest[0][0],'const':math.pow(10, xrest[0][1])}
					elif density and xtype == 'gamma':
						xrest = gamma.fit( data[i] )
						xedges = numpy.linspace( positions[0], positions[-1], 100 )
						plt.plot( xedges, gamma.pdf( xedges, xrest[0] ), linestyle = xls[j], color = self._colors[i], lw = 2 )
						xinfo['fit_gamma'] = {'a':xrest[0],'loc':xrest[1],'scale':xrest[2]}
					elif density and xtype == 'laplace':
						xrest = laplace.fit( data[i] )
						xedges = numpy.linspace( positions[0], positions[-1], 100 )
						plt.plot( xedges, laplace.pdf( xedges, xrest[0], xrest[1] ), linestyle = xls[j], color = xcolor, lw = 4 )
						xinfo['fit_laplace'] = {'loc':xrest[0],'scale':xrest[1]}
					elif density and xtype == 'cauchy':
						xrest = cauchy.fit( data[i] )
						xedges = numpy.linspace( bins_edges[0], bins_edges[-1], 100 )
						plt.plot( xedges, cauchy.pdf( xedges, xrest[0], xrest[1] ), linestyle = xls[j], color = xcolor, lw = 4 )
						xinfo['fit_cauchy'] = {'loc':xrest[0],'scale':xrest[1]}
					elif density and xtype == 'gengamma':
						xrest = gengamma.fit( data[i] )
						xedges = numpy.linspace( bins_edges[0], bins_edges[-1], 100 )
						plt.plot( xedges, gengamma.pdf( xedges, xrest[0], xrest[1], xrest[2], xrest[3] ), linestyle = xls[j], color = xcolor, lw = 4 )
						xinfo['fit_gengamma'] = {'a':xrest[0],'c':xrest[1],'loc':xrest[2],'scale':xrest[3]}
					elif density and xtype == 'chi':
						xrest = chi.fit( data[i] )
						xedges = numpy.linspace( bins_edges[0], bins_edges[-1], 100 )
						plt.plot( xedges, chi.pdf( xedges, xrest[0], xrest[1], xrest[2] ), linestyle = xls[j], color = xcolor, lw = 4 )
						xinfo['fit_chi'] = {'degree_freedom':xrest[0],'loc':xrest[1],'scale':xrest[2]}
					elif density and xtype == 'exponential':
						xrest = expon.fit( data[i] )
						xedges = numpy.linspace( bins_edges[0], bins_edges[-1], 100 )
						plt.plot( xedges, expon.pdf( xedges, xrest[0], xrest[1] ), linestyle = xls[j], color = xcolor, lw = 4 )
						xinfo['fit_exponential'] = {'loc':xrest[0],'scale':xrest[1]}
					elif density and xtype == 'pareto':
						xrest = pareto.fit( data[i] )
						xedges = numpy.linspace( bins_edges[0], bins_edges[-1], 100 )
						plt.plot( xedges, pareto.pdf( xedges, xrest[0], xrest[1], xrest[2] ), linestyle = xls[j], color = xcolor, lw = 4 )
						xinfo['fit_pareto'] = {'b':xrest[0],'loc':xrest[1],'scale':xrest[2]}
					elif density and xtype == 'powerlaw2':
						xrest = powerlaw.fit( data[i] )
						xedges = numpy.linspace( bins_edges[0], bins_edges[-1], 100 )
						plt.plot( xedges, powerlaw.pdf( xedges, xrest[0], xrest[1], xrest[2] ), linestyle = xls[j], color = xcolor, lw = 4 )
						xinfo['fit_powerlaw2'] = {'a':xrest[0],'loc':xrest[1],'scale':xrest[2]}
					elif density and xtype == 'loguniform':
						xrest = loguniform.fit( data[i] )
						xedges = numpy.linspace( bins_edges[0], bins_edges[-1], 100 )
						plt.plot( xedges, loguniform.pdf( xedges, xrest[0], xrest[1], xrest[2], xrest[3] ), linestyle = xls[j], color = xcolor, lw = 4 )
						xinfo['fit_loguniform'] = {'a':xrest[0], 'b':xrest[1],'loc':xrest[2],'scale':xrest[3]}
					elif density and xtype == 'lognorm':
						xrest = lognorm.fit( data[i] )
						xedges = numpy.linspace( bins_edges[0], bins_edges[-1], 100 )
						plt.plot( xedges, lognorm.pdf( xedges, xrest[0], xrest[1], xrest[2] ), linestyle = xls[j], color = xcolor, lw = 4 )
						xinfo['fit_lognorm'] = {'s':xrest[0], 'loc':xrest[1],'scale':xrest[2]}
					elif density and xtype == 'chi2':
						xrest = chi2.fit( data[i] )
						xedges = numpy.linspace( bins_edges[0], bins_edges[-1], 100 )
						plt.plot( xedges, chi2.pdf( xedges, xrest[0], xrest[1], xrest[2] ), linestyle = xls[j], color = xcolor, lw = 4 )
						xinfo['fit_chi2'] = {'degree_freedom':xrest[0], 'loc':xrest[1],'scale':xrest[2]}
					elif density and xtype == 'betaprime':
						xrest = betaprime.fit( data[i] )
						xedges = numpy.linspace( bins_edges[0], bins_edges[-1], 100 )
						plt.plot( xedges, betaprime.pdf( xedges, xrest[0], xrest[1], xrest[2], xrest[3] ), linestyle = xls[j], color = xcolor, lw = 4 )
						xinfo['fit_betaprime'] = {'a':xrest[0], 'b':xrest[1],'loc':xrest[2],'scale':xrest[3]}
					elif density and xtype == 'pearson3':
						xrest = pearson3.fit( data[i] )
						xedges = numpy.linspace( bins_edges[0], bins_edges[-1], 100 )
						plt.plot( xedges, pearson3.pdf( xedges, xrest[0], xrest[1], xrest[2] ), linestyle = xls[j], color = xcolor, lw = 4 )
						xinfo['fit_pearson3'] = {'skew':xrest[0], 'loc':xrest[1],'scale':xrest[2]}
					elif density and xtype == 'genpareto':
						xrest = genpareto.fit( data[i] )
						xedges = numpy.linspace( bins_edges[0], bins_edges[-1], 100 )
						plt.plot( xedges, genpareto.pdf( xedges, xrest[0], xrest[1], xrest[2] ), linestyle = xls[j], color = xcolor, lw = 4 )
						xinfo['fit_genpareto'] = {'c':xrest[0], 'loc':xrest[1],'scale':xrest[2]}
					elif density and xtype == 'ncstudent':
						xrest = ncstudent.fit( data[i] )
						xedges = numpy.linspace( bins_edges[0], bins_edges[-1], 100 )
						plt.plot( xedges, ncstudent.pdf( xedges, xrest[0], xrest[1], xrest[2], xrest[3] ), linestyle = xls[j], color = xcolor, lw = 4 )
						xinfo['fit_ncstudent'] = {'df':xrest[0], 'nc':xrest[1], 'loc':xrest[2],'scale':xrest[3]}
					elif density and xtype == 'student':
						xrest = student.fit( data[i] )
						xedges = numpy.linspace( bins_edges[0], bins_edges[-1], 100 )
						plt.plot( xedges, student.pdf( xedges, xrest[0], xrest[1], xrest[2] ), linestyle = xls[j], color = xcolor, lw = 4 )
						xinfo['fit_student'] = {'df':xrest[0], 'loc':xrest[1],'scale':xrest[2]}
					elif density and xtype == 'studentized_range':
						xrest = studentized_range.fit( data[i] )
						xedges = numpy.linspace( bins_edges[0], bins_edges[-1], 100 )
						plt.plot( xedges, studentized_range.pdf( xedges, xrest[0], xrest[1], xrest[2], xrest[3] ), linestyle = xls[j], color = xcolor, lw = 4 )
						xinfo['fit_studentized_range'] = {'k':xrest[0], 'df':xrest[1], 'loc':xrest[2],'scale':xrest[3]}

			info.append( xinfo )

			if is_multi:
				plt.xscale( xscale )
				plt.yscale( yscale )
				plt.yticks( fontsize = self._font_size )
				plt.xticks( fontsize = self._font_size )
				plt.ylabel( ylabel , fontdict = { 'size' : self._font_size })
				plt.xlabel( xlabel , fontdict = { 'size' : self._font_size })
				plt.grid( linestyle = ':' )

		if not is_multi:
			plt.xscale( xscale )
			plt.yscale( yscale )
			plt.yticks( fontsize = self._font_size )
			plt.xticks( fontsize = self._font_size )
			plt.ylabel( ylabel , fontdict = { 'size' : self._font_size })
			plt.xlabel( xlabel , fontdict = { 'size' : self._font_size })
			plt.grid( linestyle = ':' )

		if is_legend and not is_multi :
			plt.legend( frameon = False )

		if is_multi :
			plt.tight_layout()

		return info, plt
		
	def scheme_plot_fill ( self, is_fig = True, is_multi = False, is_axis_equal = False, size = (12,5), is_color = False, xdata = [], ydata = [], ystd = [], xscale = 'linear', yscale = 'linear', xlabel = '', ylabel = '', is_legend = False ):

		if is_fig :
			plt.figure( figsize = size )

		for i in range( len(xdata) ):
			if is_multi :
				plt.subplot(1, len(xdata), i+1)

			xcolor = self._colors[i]
			if is_color :
				xcolor = None
			plt.plot( xdata[i], ydata[i], color = xcolor, label = self._names[i] )
			if len(ystd) > 0 :
				plt.fill_between( xdata[i], ydata[i] - ystd[i], ydata[i] + ystd[i], color = xcolor, alpha = 0.2 )

			if is_multi :
				if is_axis_equal :
					plt.axis('equal')
				plt.xscale( xscale )
				plt.yscale( yscale )
				plt.xticks( fontsize = self._font_size )
				plt.yticks( fontsize = self._font_size )
				plt.xlabel( xlabel, fontdict = { 'size' : self._font_size } )
				plt.ylabel( ylabel, fontdict = { 'size' : self._font_size } )
				plt.grid( linestyle = ':' )

		if not is_multi:
			if is_axis_equal :
				plt.axis('equal')
			plt.xscale( xscale )
			plt.yscale( yscale )
			plt.xticks( fontsize = self._font_size )
			plt.yticks( fontsize = self._font_size )
			plt.xlabel( xlabel, fontdict = { 'size' : self._font_size } )
			plt.ylabel( ylabel, fontdict = { 'size' : self._font_size } )
			plt.grid( linestyle = ':' )
			if is_legend :
				plt.legend( frameon = False )
		else:
			plt.tight_layout()

		return plt

	def scheme_hist ( self, data = [], bins = 10, is_fit = False, type_fit = 'normal', htype = 'bar', xscale = 'linear', yscale = 'linear', density = False, xlabel = '', ylabel = '', is_multi = False, is_legend = False ):

		if not is_multi:
			plt.figure( figsize = ( 6 , 5) )
		else:
			plt.figure( figsize = ( 6*len(data) , 5) )

		info = []
		for i in range( len(data) ):
			if is_multi :
				plt.subplot(1, len(data), i+1)

			weights = None
			weights_minus = None
			bins_edges = None
			bins_edges_minus = None
			patches = None
			patches_minus = None
			xedgecolor = self._colors[i] if htype == 'step' or htype == 'flip_and_divide' else 'black' 
			if htype == 'flip_and_divide':
				weights, bins_edges, patches = plt.hist( data[i][ data[i] >= 0 ], bins = int(bins/2), density = density, color = self._colors[i], histtype = 'step', edgecolor = xedgecolor )
				weights_minus, bins_edges_minus, patches_minus = plt.hist( numpy.absolute(data[i][ data[i] < 0 ]), bins = int(bins/2), density = density, color = self._colors[i], histtype = 'step', edgecolor = xedgecolor, linestyle = ('dashed') )
			else:				
				weights, bins_edges, patches = plt.hist( data[i], bins = bins, density = density, histtype = htype, color = self._colors[i], edgecolor = xedgecolor )


			mu_data = numpy.nanmean( data[i] )
			sigma_data = math.sqrt( numpy.nanvar( data[i] ) )
			xmedian = numpy.nanmedian( data[i] )
			xindex_mode = numpy.argmax( weights )
			xmode = bins_edges[xindex_mode] + (bins_edges[xindex_mode+1] - bins_edges[xindex_mode])/2
			xkurt = kurtosis( data[i] )
			xske = skew( data[i] )
			xinfo = { 'mean' : mu_data, 'median': xmedian, 'mode':xmode, 'standar_desviation' : sigma_data, 'kurtosis' : xkurt, 'skewness' : xske }

			if is_fit:
				xtype_fit = [type_fit] if type(type_fit) is str else type_fit
				xls = ['dotted','dashed','dashdot','solid','loosely dotted','loosely dashed']
				xcolor = 'black' if is_multi else self._colors[i]
				for j, xtype in enumerate(xtype_fit):
					if density and xtype == 'normal' :
						xrest = norm.fit( data[i] )
						xedges = numpy.linspace( bins_edges[0], bins_edges[-1], 100 )
						plt.plot( xedges, norm.pdf( xedges, xrest[0], xrest[1] ), linestyle = xls[j], color = xcolor, lw = 4 )
						xinfo['fit_norm'] = {'loc':xrest[0],'scale':xrest[1]}
					elif htype != 'flip_and_divide' and xtype == 'powerlaw':
						def fit_powerlaw ( x, m, c ):
							return x*m + c
						xfit = numpy.log10(bins_edges[1::])
						yfit = numpy.log10(weights)
						i_noninf = numpy.where( ~numpy.isinf(yfit) )[0]
						xfit = xfit[i_noninf]
						yfit = yfit[i_noninf]
						xrest = curve_fit( fit_powerlaw, xfit, yfit )
						xedges = numpy.linspace( bins_edges[1], bins_edges[-1], 100 )
						plt.plot( xedges,  numpy.power(xedges, xrest[0][0] )*math.pow(10, xrest[0][1]), linestyle = xls[j], color = xcolor, lw = 4 )
						xinfo['fit_powerlaw'] = {'slope':xrest[0][0],'const':math.pow(10, xrest[0][1])}
					elif density and xtype == 'gamma':
						xrest = gamma.fit( data[i] )
						xedges = numpy.linspace( bins_edges[0], bins_edges[-1], 100 )
						plt.plot( xedges, gamma.pdf( xedges, xrest[0], xrest[1], xrest[2] ), linestyle = xls[j], color = xcolor, lw = 4 )
						xinfo['fit_gamma'] = {'a':xrest[0],'loc':xrest[1],'scale':xrest[2]}
					elif density and xtype == 'laplace':
						xrest = laplace.fit( data[i] )
						xedges = numpy.linspace( bins_edges[0], bins_edges[-1], 100 )
						plt.plot( xedges, laplace.pdf( xedges, xrest[0], xrest[1] ), linestyle = xls[j], color = xcolor, lw = 4 )
						xinfo['fit_laplace'] = {'loc':xrest[0],'scale':xrest[1]}
					elif density and xtype == 'cauchy':
						xrest = cauchy.fit( data[i] )
						xedges = numpy.linspace( bins_edges[0], bins_edges[-1], 100 )
						plt.plot( xedges, cauchy.pdf( xedges, xrest[0], xrest[1] ), linestyle = xls[j], color = xcolor, lw = 4 )
						xinfo['fit_cauchy'] = {'loc':xrest[0],'scale':xrest[1]}
					elif density and xtype == 'gengamma':
						xrest = gengamma.fit( data[i] )
						xedges = numpy.linspace( bins_edges[0], bins_edges[-1], 100 )
						plt.plot( xedges, gengamma.pdf( xedges, xrest[0], xrest[1], xrest[2], xrest[3] ), linestyle = xls[j], color = xcolor, lw = 4 )
						xinfo['fit_gengamma'] = {'a':xrest[0],'c':xrest[1],'loc':xrest[2],'scale':xrest[3]}
					elif density and xtype == 'chi':
						xrest = chi.fit( data[i] )
						xedges = numpy.linspace( bins_edges[0], bins_edges[-1], 100 )
						plt.plot( xedges, chi.pdf( xedges, xrest[0], xrest[1], xrest[2] ), linestyle = xls[j], color = xcolor, lw = 4 )
						xinfo['fit_chi'] = {'degree_freedom':xrest[0],'loc':xrest[1],'scale':xrest[2]}
					elif density and xtype == 'exponential':
						xrest = expon.fit( data[i] )
						xedges = numpy.linspace( bins_edges[0], bins_edges[-1], 100 )
						plt.plot( xedges, expon.pdf( xedges, xrest[0], xrest[1] ), linestyle = xls[j], color = xcolor, lw = 4 )
						xinfo['fit_exponential'] = {'loc':xrest[0],'scale':xrest[1]}
					elif density and xtype == 'pareto':
						xrest = pareto.fit( data[i] )
						xedges = numpy.linspace( bins_edges[0], bins_edges[-1], 100 )
						plt.plot( xedges, pareto.pdf( xedges, xrest[0], xrest[1], xrest[2] ), linestyle = xls[j], color = xcolor, lw = 4 )
						xinfo['fit_pareto'] = {'b':xrest[0],'loc':xrest[1],'scale':xrest[2]}
					elif density and xtype == 'powerlaw2':
						xrest = powerlaw.fit( data[i] )
						xedges = numpy.linspace( bins_edges[0], bins_edges[-1], 100 )
						plt.plot( xedges, powerlaw.pdf( xedges, xrest[0], xrest[1], xrest[2] ), linestyle = xls[j], color = xcolor, lw = 4 )
						xinfo['fit_powerlaw2'] = {'a':xrest[0],'loc':xrest[1],'scale':xrest[2]}
					elif density and xtype == 'loguniform':
						xrest = loguniform.fit( data[i] )
						xedges = numpy.linspace( bins_edges[0], bins_edges[-1], 100 )
						plt.plot( xedges, loguniform.pdf( xedges, xrest[0], xrest[1], xrest[2], xrest[3] ), linestyle = xls[j], color = xcolor, lw = 4 )
						xinfo['fit_loguniform'] = {'a':xrest[0], 'b':xrest[1],'loc':xrest[2],'scale':xrest[3]}
					elif density and xtype == 'lognorm':
						xrest = lognorm.fit( data[i] )
						xedges = numpy.linspace( bins_edges[0], bins_edges[-1], 100 )
						plt.plot( xedges, lognorm.pdf( xedges, xrest[0], xrest[1], xrest[2] ), linestyle = xls[j], color = xcolor, lw = 4 )
						xinfo['fit_lognorm'] = {'s':xrest[0], 'loc':xrest[1],'scale':xrest[2]}
					elif density and xtype == 'chi2':
						xrest = chi2.fit( data[i] )
						xedges = numpy.linspace( bins_edges[0], bins_edges[-1], 100 )
						plt.plot( xedges, chi2.pdf( xedges, xrest[0], xrest[1], xrest[2] ), linestyle = xls[j], color = xcolor, lw = 4 )
						xinfo['fit_chi2'] = {'degree_freedom':xrest[0], 'loc':xrest[1],'scale':xrest[2]}
					elif density and xtype == 'betaprime':
						xrest = betaprime.fit( data[i] )
						xedges = numpy.linspace( bins_edges[0], bins_edges[-1], 100 )
						plt.plot( xedges, betaprime.pdf( xedges, xrest[0], xrest[1], xrest[2], xrest[3] ), linestyle = xls[j], color = xcolor, lw = 4 )
						xinfo['fit_betaprime'] = {'a':xrest[0], 'b':xrest[1],'loc':xrest[2],'scale':xrest[3]}
					elif density and xtype == 'pearson3':
						xrest = pearson3.fit( data[i] )
						xedges = numpy.linspace( bins_edges[0], bins_edges[-1], 100 )
						plt.plot( xedges, pearson3.pdf( xedges, xrest[0], xrest[1], xrest[2] ), linestyle = xls[j], color = xcolor, lw = 4 )
						xinfo['fit_pearson3'] = {'skew':xrest[0], 'loc':xrest[1],'scale':xrest[2]}
					elif density and xtype == 'genpareto':
						xrest = genpareto.fit( data[i] )
						xedges = numpy.linspace( bins_edges[0], bins_edges[-1], 100 )
						plt.plot( xedges, genpareto.pdf( xedges, xrest[0], xrest[1], xrest[2] ), linestyle = xls[j], color = xcolor, lw = 4 )
						xinfo['fit_genpareto'] = {'c':xrest[0], 'loc':xrest[1],'scale':xrest[2]}
					elif density and xtype == 'ncstudent':
						xrest = ncstudent.fit( data[i] )
						xedges = numpy.linspace( bins_edges[0], bins_edges[-1], 100 )
						plt.plot( xedges, ncstudent.pdf( xedges, xrest[0], xrest[1], xrest[2], xrest[3] ), linestyle = xls[j], color = xcolor, lw = 4 )
						xinfo['fit_ncstudent'] = {'df':xrest[0], 'nc':xrest[1], 'loc':xrest[2],'scale':xrest[3]}
					elif density and xtype == 'student':
						xrest = student.fit( data[i] )
						xedges = numpy.linspace( bins_edges[0], bins_edges[-1], 100 )
						plt.plot( xedges, student.pdf( xedges, xrest[0], xrest[1], xrest[2] ), linestyle = xls[j], color = xcolor, lw = 4 )
						xinfo['fit_student'] = {'df':xrest[0], 'loc':xrest[1],'scale':xrest[2]}
					elif density and xtype == 'studentized_range':
						xrest = studentized_range.fit( data[i] )
						xedges = numpy.linspace( bins_edges[0], bins_edges[-1], 100 )
						plt.plot( xedges, studentized_range.pdf( xedges, xrest[0], xrest[1], xrest[2], xrest[3] ), linestyle = xls[j], color = xcolor, lw = 4 )
						xinfo['fit_studentized_range'] = {'k':xrest[0], 'df':xrest[1], 'loc':xrest[2],'scale':xrest[3]}
					elif xtype == 'powerlaw' and htype == 'flip_and_divide':
						def fit_powerlaw ( x, m, c ):
							return x*m + c
						xfit = numpy.log10(bins_edges[1::])
						yfit = numpy.log10(weights)
						i_noninf = numpy.where( ~numpy.isinf(yfit) )[0]
						xfit = xfit[i_noninf]
						yfit = yfit[i_noninf]
						xrest = curve_fit( fit_powerlaw, xfit, yfit )
						xedges = numpy.linspace( bins_edges[1], bins_edges[-1], 100 )
						plt.plot( xedges,  numpy.power(xedges, xrest[0][0] )*math.pow(10, xrest[0][1]), linestyle = '-', color = xcolor, lw = 4 )
						
						xfit_minus = numpy.log10(bins_edges_minus[1::])
						yfit_minus = numpy.log10(weights_minus)
						i_noninf_minus = numpy.where( ~numpy.isinf(yfit_minus) )[0]
						xfit_minus = xfit_minus[i_noninf_minus]
						yfit_minus = yfit_minus[i_noninf_minus]
						xrest_minus = curve_fit( fit_powerlaw, xfit_minus, yfit_minus )
						xedges_minus = numpy.linspace( bins_edges_minus[1], bins_edges_minus[-1], 100 )
						plt.plot( xedges_minus,  numpy.power(xedges_minus, xrest_minus[0][0] )*math.pow(10, xrest_minus[0][1]), linestyle = ':', color = xcolor, lw = 4 )
						
						intersection_x = ( math.pow(10, xrest[0][1]) - math.pow(10, xrest_minus[0][1]) )/(xrest_minus[0][0] - xrest[0][0])
						intersection_y = ( xrest[0][0]*intersection_x + math.pow(10, xrest[0][1]) )

						xinfo['fit_powerlaw'] = {'slope':xrest[0][0],'const':math.pow(10, xrest[0][1]),'slope_minus':xrest_minus[0][0],'const_minus':math.pow(10, xrest_minus[0][1]),'intersection': ( intersection_x, intersection_y ) }

					
			info.append( xinfo )

			if is_multi:
				plt.xscale( xscale )
				plt.yscale( yscale )
				plt.yticks( fontsize = self._font_size )
				plt.xticks( fontsize = self._font_size )
				plt.ylabel( ylabel , fontdict = { 'size' : self._font_size })
				plt.xlabel( xlabel , fontdict = { 'size' : self._font_size })
				plt.grid( linestyle = ':' )

		if not is_multi:
			plt.xscale( xscale )
			plt.yscale( yscale )
			plt.yticks( fontsize = self._font_size )
			plt.xticks( fontsize = self._font_size )
			plt.ylabel( ylabel , fontdict = { 'size' : self._font_size })
			plt.xlabel( xlabel , fontdict = { 'size' : self._font_size })
			plt.grid( linestyle = ':' )

		if is_legend and not is_multi :

			legend = []
			for i in range( len(data) ):
				legend.append( Line2D([0], [0], color = self._colors[i] ) )
			
			plt.legend( legend, self._names, loc = 'upper left', frameon = False )

		if is_multi :
			plt.tight_layout()

		return info, plt
		
	def scheme_single_boxplot ( self, data = [], ylabel = '', span = 0.2, is_test = False, showfliers = True, spancap = 2.1 ):

		index_position = []
		plt.figure( figsize = ( 1.5*len(data) , 5) )

		for i in range( len(data) ):
			plt.boxplot( x = [ data[i] ], positions = [ 1 + span*i ], widths=0.1, notch = True, patch_artist = True, showfliers = showfliers, boxprops = dict( facecolor = self._colors[i] ), medianprops = dict( linewidth = 1, color = 'black' ) )
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

	def scheme_multiple_boxplot ( self, data = [], xlabels = [], color_box = [], span = 0.2, ylabel = '', showfliers = True, is_test = False, is_legend = False, label_legend = [], spancap = 2.1 ):

		plt.figure( figsize = (1.5*len(data[0])*len(data), 5) )

		index_position  = [ ]
		xticks_position = []
		for i in range( len(data) ):
			positions = []
			for j in range( len(data[i]) ):
				positions.append( j + 1 + span*i )
			index_position.append( positions )
			xcolor = color_box[i] if len(color_box) > 0 else self._colors[i]
			plt.boxplot( x = data[i], positions = positions, widths=0.1, notch = True, patch_artist = True, showfliers = showfliers, boxprops = dict( facecolor = xcolor ), medianprops = dict( linewidth = 1, color = 'black' ) )

		for i in range( len(index_position[0]) ):
			tickinter = abs(index_position[ len(index_position) - 1 ][0] - index_position[0][0])
			xticks_position.append( index_position[0][i] + tickinter/2 )

		plt.xticks( xticks_position, xlabels, fontsize = self._font_size )
		plt.yticks( fontsize = self._font_size )
		plt.ylabel( ylabel , fontdict = { 'size' : self._font_size })
		plt.grid( linestyle = ':' )
		
		if is_legend :

			legend = []
			for i in range( len(data) ):
				xcolor = color_box[i] if len(color_box) > 0 else self._colors[i]
				legend.append( Line2D([0], [0], color = xcolor ) )
			
			xnames = label_legend if len(label_legend) > 0 else self._names
			plt.legend( legend, xnames, loc = 'upper right', frameon = False )

		if is_test :
			pvalues = []
			mpair= []
			count_true_test = []
			for i in range( len(data[0]) ):
				mcomb = numpy.nonzero( numpy.tri( len(data) , len(data)  , -1) )
				xmpair = numpy.dstack( (mcomb[1],mcomb[0]) )[0]
				mpair.append(xmpair)
				xpval = []
				xcount = 0
				for row in xmpair:
					test_pvalue = stats.ttest_ind( data[ row[0] ][i], data[ row[1] ][i] ).pvalue
					xpval.append( test_pvalue )
					if test_pvalue < self._test_threshold_value:
						xcount = xcount + 1
				pvalues.append( xpval )
				count_true_test.append( xcount )

			if numpy.amax(count_true_test) > 0:

				lims = plt.axis()
				y_inter = ( lims[3] - lims[2] )/self._interval_axis_plot
				plt.ylim([ lims[2], lims[3] + y_inter ])

				spany = y_inter/numpy.amax(count_true_test)
				spanx = span/2
				for i in range( len( pvalues ) ):
					for j in range( len(pvalues[i]) ):
						if pvalues[i][j] < self._test_threshold_value:
							difspancap = abs( mpair[i][j][0] - mpair[i][j][1] )

							plt.annotate('**', xy=(i + 1 + span*mpair[i][j][0] + spanx*difspancap , lims[3] + y_inter - spany*j), ha='center', va='bottom', xycoords = 'data', arrowprops=dict(arrowstyle='-[, widthB='+str( spancap*difspancap )+', lengthB=0.5') )


		return plt





