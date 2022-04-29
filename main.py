import matplotlib.pyplot as plt
import numpy as np
import pandas
import PIL

energies = ['80000','120000','200000','300000']
a02 = 2.8002852E-21

def normalize(x,y,total):
    print(x.shape)
    return y*np.sin(np.radians(x)) * total / np.sum(np.diff(x)*np.convolve(y*np.sin(np.radians(x)),
                                                                           [0.5, 0.5],
                                                                           mode='valid'))

def plotnist(ax,x,y,color):
    return ax.plot(x,
            2*np.pi*np.pi/180*np.sin(np.radians(x))*y,
            color=color)

for energy in energies:
    os_data = pandas.read_csv('DCS_Os_'+energy+'_eV.csv',skiprows=9)
    print(2*np.pi*np.pi/180*np.sum(np.sin(np.radians(os_data.to_numpy()[1:, 0])) * os_data.to_numpy()[1:, 1] * np.diff(os_data.to_numpy()[:, 0])))

    c_data = pandas.read_csv('DCS_C_'+energy+'_eV.csv',skiprows=9)
    fig, ax1 = plt.subplots()
    # plt.plot(os_data.to_numpy()[:, 0], normalize(os_data.to_numpy()[:, 0],os_data.to_numpy()[:, 1],a02*8.668275E-1))
    color = 'tab:red'
    ax1.set_xlabel('angle [$degree$]')
    ax1.set_ylabel('cross section [${a_{0}}^{2}/degree$]', color=color)
    ln1= plotnist(ax1,
             os_data.to_numpy()[:, 0],
             os_data.to_numpy()[:, 1],
             color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('cross section [${a_{0}}^{2}/degree$]', color=color)  # we already handled the x-label with ax1
    ln2 = plotnist(ax2,
             c_data.to_numpy()[:, 0],
             c_data.to_numpy()[:, 1],
             color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.xlim([0, 5])
    plt.title(energy + ' keV')
    plt.xlabel('angle [degree]')
    lns = ln1 + ln2
    ax1.legend(lns, ['Os','C'], loc=0)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    print(int(energy)*os_data.to_numpy()[:,0][np.argmax(os_data.to_numpy()[:,0]*os_data.to_numpy()[:,1])])
    print(sum((os_data.to_numpy()[:, 0]*os_data.to_numpy()[:, 1])[np.logical_and(os_data.to_numpy()[:, 0]>0, os_data.to_numpy()[:, 0]<1.5)])/sum(os_data.to_numpy()[:, 0]*os_data.to_numpy()[:, 1]))
stopping = pandas.read_csv('edata.pl.txt',skiprows=6)

plt.figure()

plt.plot(stopping.to_numpy()[:, 0].T, stopping.to_numpy()[:, 1].T)
plt.xscale('log')
plt.show()


im = PIL.Image.open(r"C(0-0450eV).tif")
a = np.array(im)
#plt.plot(np.log(a.T))
#plt.ylim([0,10_000])
#plt.xlim([80,114])
print(np.argwhere(a==np.max(a)))

plt.figure()
plt.plot(np.arange(start=-57,stop=455,step=0.5),
         a[0,:])
plt.yscale('log')
plt.xlim([-5,400])
plt.show()
limit=2
print(np.sum(a[0,114-limit:115+limit]))
print(np.sum(a))
print(a.shape)
print(np.sum(np.arange(start=-57,stop=455,step=0.5)*a)/np.sum(a))
print(1/(2.476E6*2/10_000_000/178.145))