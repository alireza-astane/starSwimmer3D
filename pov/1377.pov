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
    sphere { m*<0.5200232049580428,-4.630421408504798e-18,0.9999340780567927>, 1 }        
    sphere {  m*<0.5964240385327086,-4.730904884280471e-19,3.9989636616618895>, 1 }
    sphere {  m*<7.3887175514189956,3.3469352183403244e-18,-1.6273233535373863>, 1 }
    sphere {  m*<-4.277386955783217,8.164965809277259,-2.212268687040094>, 1}
    sphere { m*<-4.277386955783217,-8.164965809277259,-2.2122686870400976>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5964240385327086,-4.730904884280471e-19,3.9989636616618895>, <0.5200232049580428,-4.630421408504798e-18,0.9999340780567927>, 0.5 }
    cylinder { m*<7.3887175514189956,3.3469352183403244e-18,-1.6273233535373863>, <0.5200232049580428,-4.630421408504798e-18,0.9999340780567927>, 0.5}
    cylinder { m*<-4.277386955783217,8.164965809277259,-2.212268687040094>, <0.5200232049580428,-4.630421408504798e-18,0.9999340780567927>, 0.5 }
    cylinder {  m*<-4.277386955783217,-8.164965809277259,-2.2122686870400976>, <0.5200232049580428,-4.630421408504798e-18,0.9999340780567927>, 0.5}

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
    sphere { m*<0.5200232049580428,-4.630421408504798e-18,0.9999340780567927>, 1 }        
    sphere {  m*<0.5964240385327086,-4.730904884280471e-19,3.9989636616618895>, 1 }
    sphere {  m*<7.3887175514189956,3.3469352183403244e-18,-1.6273233535373863>, 1 }
    sphere {  m*<-4.277386955783217,8.164965809277259,-2.212268687040094>, 1}
    sphere { m*<-4.277386955783217,-8.164965809277259,-2.2122686870400976>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5964240385327086,-4.730904884280471e-19,3.9989636616618895>, <0.5200232049580428,-4.630421408504798e-18,0.9999340780567927>, 0.5 }
    cylinder { m*<7.3887175514189956,3.3469352183403244e-18,-1.6273233535373863>, <0.5200232049580428,-4.630421408504798e-18,0.9999340780567927>, 0.5}
    cylinder { m*<-4.277386955783217,8.164965809277259,-2.212268687040094>, <0.5200232049580428,-4.630421408504798e-18,0.9999340780567927>, 0.5 }
    cylinder {  m*<-4.277386955783217,-8.164965809277259,-2.2122686870400976>, <0.5200232049580428,-4.630421408504798e-18,0.9999340780567927>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    