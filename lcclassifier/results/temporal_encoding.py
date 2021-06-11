from __future__ import print_function
from __future__ import division
from . import C_

###################################################################################################################################################

def plot_temporal_encoding(rootdir, cfilename, kf, lcset_name, model_names,
	train_mode='pre-training',
	layers=3,
	figsize=(12,10),
	n=1e3,
	):
	for kmn,model_name in enumerate(model_names):
		load_roodir = f'{rootdir}/{model_name}/{train_mode}/temporal_encoding/{cfilename}/{kf}@{lcset_name}'
		files, files_ids = fcfiles.gather_files_by_id(load_roodir, fext='d')
		print(f'{model_name} {files_ids}({len(files_ids)}#)')
		if len(files)==0:
			continue

		survey = files[0]()['survey']
		band_names = files[0]()['band_names']
		class_names = files[0]()['class_names']
		mn_dict = strings.get_dict_from_string(model_name)
		mdl = mn_dict['mdl']
		is_parallel = 'Parallel' in mdl
		days = files[0]()['days']
		#days = np.linspace(days[0], days[-1], int(n))
		#days = np.linspace(0, 40, int(n))

		def get_fourier(t, weights, te_periods, te_phases):
			x = np.zeros_like(t)
			for kw,w in enumerate(weights):
				x += w*np.sin(2*np.pi*t/te_periods[kw]+te_phases[kw])
			return x

		###
		b = 'r'
		fig, axs = plt.subplots(layers, 1, figsize=figsize)
		cmap = cm.get_cmap('viridis', 100)

		for layer in range(0, layers):
			ax = axs[layer]
			#for kfile,file in enumerate(files):
			kfile = 1
			d = files[kfile]()['temporal_encoding_info']['encoder'][f'ml_attn.{b}' if is_parallel else f'ml_attn']['te_film'][layer]
			weight = d['weight']
			alpha_weights, beta_weights = np.split(weight.T, 2, axis=-1)
			#kfu = 1
			dalphas = []
			alphas = []
			dbetas = []
			betas = []
			for kfu in range(0, len(weight)//2):
				te_ws = d['te_ws']
				te_periods = d['te_periods']
				#print(te_periods)
				te_phases = d['te_phases']
				#print(alpha_weight.shape, beta_weight.shape, te_ws.shape, te_phases.shape)

				alpha = get_fourier(days, alpha_weights[:,kfu], te_periods, te_phases)
				alphas += [(alpha[None]*1e0)**2] # fixme luego usar (alpha+1)
				dalphas += [(np.diff(alpha, prepend=alpha[0])[None]*1e0)**2]

				beta = get_fourier(days, beta_weights[:,kfu], te_periods, te_phases)
				betas += [(beta[None]*1e0)**2]
				dbetas += [(np.diff(beta, prepend=beta[0])[None]*1e0)**2]
				#ax.plot(days, alpha, 'r', lw=1, label=f'scale learned curves' if kfu==0 else None)#, c=cmap(k/len(te_ws)))
				#ax.plot(days, beta, 'g', lw=1, label='bias learned curves' if kfu==0 else None)#, c=cmap(k/len(te_ws)))
				ax.grid(alpha=0.5)
				ax.legend()
				label = f'encoder-layer={layer} - band={b}' if is_parallel else f'encoder-layer={layer}'
				ax.set_ylabel(label)
				#ax.set_yticklabels([])
				#ax.set_xlim([0,1000])

			ax.plot(days, np.concatenate(alphas, axis=0).mean(axis=0), '-r', lw=1, label=f'scale learned curves' if kfu==0 else None)#, c=cmap(k/len(te_ws)))
			ax.plot(days, np.concatenate(betas, axis=0).mean(axis=0), '-g', lw=1, label='bias learned curves' if kfu==0 else None)#, c=cmap(k/len(te_ws)))
			ax.plot(days, np.concatenate(dalphas, axis=0).mean(axis=0), '--r', lw=1, label=f'scale learned curves' if kfu==0 else None)#, c=cmap(k/len(te_ws)))
			ax.plot(days, np.concatenate(dbetas, axis=0).mean(axis=0), '--g', lw=1, label='bias learned curves' if kfu==0 else None)#, c=cmap(k/len(te_ws)))
			ax.set_xlabel('time [days]')

		_label = strings.get_string_from_dict({k:mn_dict[k] for k in mn_dict.keys() if k in label_keys}, key_key_separator=' - ')
		label = f'{mdl} ({_label})'
		title = ''
		title += f'{label}'+'\n'
		title += f'survey={survey} [{kf}@{lcset_name}] - bands={"".join(band_names)}'+'\n'
		title += f'train-mode={train_mode}'+'\n'
		axs[0].set_title(title[:-1])
		plt.show()