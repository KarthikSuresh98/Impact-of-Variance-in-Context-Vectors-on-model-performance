import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
CMAP = sns.light_palette("blue")

def scatter_plot(df, task = 'qa', eval_metric = 'var'):
    plt.figure(figsize=(10,10))
    sns.set_theme(style="whitegrid")

    if eval_metric == 'var':
        plot = sns.lmplot(x='Variance', y='F1 Score', data=df, hue='Model', fit_reg=False)
        plot.set(xlabel = "Variance across the context vectors")
    else:
        plot = sns.lmplot(x='CSE', y='F1 Score', data=df, hue='Model', fit_reg=False)
        plot.set(xlabel = "Cosine Similarity Error (CSE)")

    plot.figure.savefig(task + "_f1_" + eval_metric  + ".png", dpi=120)

def plt_all_attentions(attentions, file_name):
  num_heads = attentions.shape[0]

  fig = plt.figure(figsize=(50,20),facecolor='w', edgecolor='k')
  plt.tight_layout()
  fig.subplots_adjust(hspace=.2, wspace=0)

  #loop through all attention heads
  for i in range(0, num_heads):
      ax = fig.add_subplot(2, num_heads/2, i+1)
      ax.set_title("A{}".format(i))
      
      #get the matrix for particular layer and current attention head
      mat = attentions[i]

      #plot heatmap using sns
      res = sns.heatmap(mat.detach().numpy(),cmap=CMAP, ax=ax, cbar=False, yticklabels=[],xticklabels=[])
      
      # make frame visible
      for _, spine in res.spines.items():
          spine.set_visible(True)
  fig.savefig(file_name)

def plt_attentions(mat, labs, file_name, fig_size=(5,5), annot=False, cmap = CMAP, title=None):
  '''
  plot the NxN matrix passed as a heat map
  
  mat: square matrix to visualize
  labs: labels for xticks and yticks (the tokens in our case)
  '''
  fig, ax = plt.subplots(figsize=fig_size) 
  ax = sns.heatmap(mat.detach().numpy(),cmap=CMAP, cbar=False, yticklabels=[],xticklabels=[])
  if title:
    ax.set_title(title)
  fig.savefig(file_name)