




angle1 = 60.0

rad = 0.2
thick = 0.025 ; 0.05


l1 =  !PI* angle1/180.0 * rad
l2 =  !PI * (thick/2)
l3 =  !PI* angle1/180.0 * (rad-thick)
l4 =  !PI * (thick/2)

L = l1+l2+l3+l4


N = 75


off = L/N

pos = dblarr(3, 2*N)

for i=0, N-1 do begin
   dl = L/N * i

   if dl lt l1 then begin
      alpha = -dl/l1 * angle1 - ( 90 - angle1/2)
      x = (rad+off/2)*cos(alpha * !PI/180)
      y = (rad+off/2)*sin(alpha * !PI/180)

      pos(0,i) = x
      pos(1,i) = y

      x = (rad-off/2)*cos(alpha * !PI/180)
      y = (rad-off/2)*sin(alpha * !PI/180)

      pos(0,i+N) = x
      pos(1,i+N) = y

   endif else begin
      if dl lt (l1+l2) then begin
         alpha = -(dl-l1)/l2 * 180.0  - angle1/2 - 90
         x = ((thick+off)/2)*cos(alpha * !PI/180)
         y = ((thick+off)/2)*sin(alpha * !PI/180)

         x1 = (rad+off/2-(thick+off)/2)*cos((-angle1 - 90 + angle1/2) * !PI/180)
         y1 = (rad+off/2-(thick+off)/2)*sin((-angle1 - 90 + angle1/2) * !PI/180)
         x += x1
         y += y1

         pos(0,i) = x
         pos(1,i) = y


         x = ((thick-off)/2)*cos(alpha * !PI/180)
         y = ((thick-off)/2)*sin(alpha * !PI/180)

         x += (rad-off/2-(thick-off)/2)*cos((-angle1 - 90 + angle1/2) * !PI/180)
         y += (rad-off/2-(thick-off)/2)*sin((-angle1 - 90 + angle1/2) * !PI/180)

         pos(0,i+N) = x
         pos(1,i+N) = y


      endif else begin
         if dl lt l1+l2+l3 then begin

            alpha = (dl-(l1+l2))/l3 * angle1  - angle1/2 - 90
            x = (rad+off/2-(thick+off))*cos(alpha * !PI/180)
            y = (rad+off/2-(thick+off))*sin(alpha * !PI/180)
            pos(0,i) = x
            pos(1,i) = y

            alpha = (dl-(l1+l2))/l3 * angle1  - angle1/2 - 90
            x = (rad-off/2-(thick-off))*cos(alpha * !PI/180)
            y = (rad-off/2-(thick-off))*sin(alpha * !PI/180)
            pos(0,i+N) = x
            pos(1,i+N) = y

          endif else begin

            alpha = -(dl-(l1+l2+l3))/l4 * 180 +  angle1/2 + 90
            x = ((thick+off)/2)*cos(alpha * !PI/180)
            y = ((thick+off)/2)*sin(alpha * !PI/180)

            x2 = (rad+off/2-(thick+off)/2)*cos(( - 90 + angle1/2) * !PI/180)
            y2 = (rad+off/2-(thick+off)/2)*sin(( - 90 + angle1/2) * !PI/180)

            x += x2
            y += y2

            pos(0,i) = x
            pos(1,i) = y


            alpha = -(dl-(l1+l2+l3))/l4 * 180 +  angle1/2 + 90
            x = ((thick-off)/2)*cos(alpha * !PI/180)
            y = ((thick-off)/2)*sin(alpha * !PI/180)
            x += (rad-off/2-(thick-off)/2)*cos(( - 90 + angle1/2) * !PI/180)
            y += (rad-off/2-(thick-off)/2)*sin(( - 90 + angle1/2) * !PI/180)

            pos(0,i+N) = x
            pos(1,i+N) = y



         endelse
      endelse
   endelse


endfor

xc = total(pos(0,*))/n_elements(pos(0,*))
yc = total(pos(1,*))/n_elements(pos(1,*))

dx = 0.5+0.25 - xc
dy = 0.5 - yc


pos(0,*) += dx
pos(1,*) += dy




NgX = 100L
NgY = 100L

Ntot = NgX*NgY

rho0 = 1.0
P0 = 3.0/5.0D

gamma = 5.0/3D

cs = sqrt(gamma * P0/rho0)

print, "cs= ", cs

BoxX = 1.0D
BoxY = 1.0D 


print,"BoxX=",BoxX, "  BoxY=", BoxY

pos1 = dblarr(3, Ntot)

ip = 0L

for j = 0L, NgY - 1 do begin
   for i = 0L, NgX - 1 do begin
      pos1(0, ip) = BoxX * (i+0.5D)/NgX
      pos1(1, ip) = BoxY * (j+0.5D)/NgY
      ip++
   endfor
endfor



phi = atan(pos1(1,*)-dy, pos1(0,*)-dx)
r = sqrt((pos1(0,*)-dx)^2 + (pos1(1,*)-dy)^2)

ind = where(((r gt rad-thick) and (r lt rad) and $
             (phi gt -!PI+ (90-angle1/2)/180.0*!PI) $
             and (phi lt -(90-angle1/2)/180.0*!PI)))

; $ or
;            

ind = where(((sqrt((pos1(0,*)-dx-x1)^2 + (pos1(1,*)-dy-y1)^2) lt thick/2+1.5*off) or $
            (sqrt((pos1(0,*)-dx-x2)^2 + (pos1(1,*)-dy-y2)^2) lt thick/2+1.5*off) or $
      ( (r gt rad-thick-1.5*off) and (r lt rad+1.5*off) and $
             (phi gt -!PI+ (90-angle1/2)/180.0*!PI)  $
             and (phi lt -(90-angle1/2)/180.0*!PI))))



pos1(0,ind) = -1.0e30

ind = where(pos1(0,*) ge 0)

x = pos1(0,ind)
y = pos1(1,ind)



plot, x, y, psym=4, xrange=[0.6,1.0], yrange=[0.3,0.7]

NN = n_elements(x)


;plot, pos1(0,0:NgX*NgY-1), pos1(1,0:NgX*NgY-1), psym=3



oplot, pos(0,0:N-1), pos(1,0:N-1), psym=5
oplot, pos(0,N:*), pos(1,N:*), psym=6, color=255



PP = dblarr(3, NN+2*N)
VV = dblarr(3, NN+2*N)
rho=  dblarr(NN+2*N)
press= dblarr(NN+2*N)
id = lonarr(NN+2*N)


PP(0,0:NN-1) = x(*)
PP(1,0:NN-1) = y(*)

PP(0,NN:*) = pos(0,*)
PP(1,NN:*) = pos(1,*)




id(0:NN-1) = lindgen(NN)+1
id(NN:NN+N-1) = lindgen(N)+10000000
id(NN+N:NN+2*N-1) = lindgen(N)+20000000


P0 = 1.0D
press(*) = P0 


rho(*) = 1.0

ind = where(PP(1,*) gt 0.6)
rho(ind) = 0.5

ind = where(id ge 20000000)
rho(ind) = 5.0

conservedtracer = rho

  npart=lonarr(6)
  massarr=dblarr(6)
  time=0.0D
  redshift=0.0D
  flag_sfr=0L
  flag_feedback=0L
  npartTotal=lonarr(6)
  flag_cooling=0L
  num_files = 1L
  BoxSize=double(boxX)
  Omega0=0.0D
  OmegaLambda=0.0D
  HubbleParam=0.0D
  flag_stellarage=0L
  flag_metals=0L
  highwork=lonarr(6)
  flag_entropy=0L
  flag_double=1L
  flag_lpt=0L
  factor=0.0
  la=intarr(48/2)

  npart(0) = NN+2*N
  npartTotal(0) = NN+2*N

  u = dblarr(NN+2*N)
  u(*) = press / (gamma-1) / rho

  openw, 1, "ics.dat", /f77_unformatted

  writeu,1,npart,massarr,time,redshift,flag_sfr,flag_feedback,npartTotal, $
         flag_cooling, num_files, BoxSize, Omega0, OmegaLambda, HubbleParam, flag_stellarage, flag_metals, highwork,$
         flag_entropy, flag_double, flag_lpt, factor, la
  writeu,1,PP
  writeu,1,VV
  writeu,1,id
  writeu,1,rho
  ;writeu,1,conservedtracer
  writeu,1,u
  close,1

  r = max(sqrt((pos(0,*)-0.5)^2 + (pos(1,*)-0.5)^2))

print,"rmax =", r

omega = 2 * !PI / 5 ; angular velocity 

dt = 0.5 * off / (omega * r)

print, "max dt=", dt

ende:
end
