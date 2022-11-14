# OpticalFlow3D
This is the 3d optical flow for volume. I made this for the particle flow reconstruction.


The function optical_flow(o) is the function to compute the flow. return flow is 3d dimension tensor. Mean the flow in x, y,z dimension. 
input o is 4d dimension tensor. The first 3 dimensions are the x,y,z. The last is the frame. For my default case i only compute 2 frame. So o is a dimension like [128,128,100.2].

The follow is a small example:

```
data1=np.zeros((128,128,100))
data2=np.zeros((128,128,100))

data1[10:15,10:15,10]=1

data2[12:17,12:17,10]=1

o=np.stack((data1,data2),axis=-1)
[xx, yy, zz] = np.meshgrid(np.arange(o.shape[1]),np.arange(o.shape[0]),np.arange(o.shape[2]))


o1_warp_t=interp3d( xx + flow[:, :, :, 1],yy + flow[:, :, :, 0],zz + flow[:, :, :, 2],o[:, :, :, 0])

plt.figure()
plt.imshow(o[:,:,10,0])
plt.show()
plt.imshow(o1_warp_t[:,:,10])
plt.show()
plt.imshow(o[:,:,10,1])
plt.show()
```

if there is any question it is welcomed to contact me.
