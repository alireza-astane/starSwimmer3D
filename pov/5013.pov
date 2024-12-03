#version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-0.2827962639821849,-0.13881251655549,-1.6686384279242383>, 1 }        
    sphere {  m*<0.5287005923854455,0.2906867184537932,8.289123679832676>, 1 }
    sphere {  m*<2.5401001978361344,-0.03360711335844163,-2.9405623594634114>, 1 }
    sphere {  m*<-1.9052125766936197,2.189670045401937,-2.641228654705865>, 1}
    sphere { m*<-1.6374253556557878,-2.6980218970019605,-2.4516823695432945>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5287005923854455,0.2906867184537932,8.289123679832676>, <-0.2827962639821849,-0.13881251655549,-1.6686384279242383>, 0.5 }
    cylinder { m*<2.5401001978361344,-0.03360711335844163,-2.9405623594634114>, <-0.2827962639821849,-0.13881251655549,-1.6686384279242383>, 0.5}
    cylinder { m*<-1.9052125766936197,2.189670045401937,-2.641228654705865>, <-0.2827962639821849,-0.13881251655549,-1.6686384279242383>, 0.5 }
    cylinder {  m*<-1.6374253556557878,-2.6980218970019605,-2.4516823695432945>, <-0.2827962639821849,-0.13881251655549,-1.6686384279242383>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    #version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-0.2827962639821849,-0.13881251655549,-1.6686384279242383>, 1 }        
    sphere {  m*<0.5287005923854455,0.2906867184537932,8.289123679832676>, 1 }
    sphere {  m*<2.5401001978361344,-0.03360711335844163,-2.9405623594634114>, 1 }
    sphere {  m*<-1.9052125766936197,2.189670045401937,-2.641228654705865>, 1}
    sphere { m*<-1.6374253556557878,-2.6980218970019605,-2.4516823695432945>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5287005923854455,0.2906867184537932,8.289123679832676>, <-0.2827962639821849,-0.13881251655549,-1.6686384279242383>, 0.5 }
    cylinder { m*<2.5401001978361344,-0.03360711335844163,-2.9405623594634114>, <-0.2827962639821849,-0.13881251655549,-1.6686384279242383>, 0.5}
    cylinder { m*<-1.9052125766936197,2.189670045401937,-2.641228654705865>, <-0.2827962639821849,-0.13881251655549,-1.6686384279242383>, 0.5 }
    cylinder {  m*<-1.6374253556557878,-2.6980218970019605,-2.4516823695432945>, <-0.2827962639821849,-0.13881251655549,-1.6686384279242383>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    