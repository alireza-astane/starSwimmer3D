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
    sphere { m*<-2.2608489296671996e-18,-2.4983699987804424e-18,0.5874038446480405>, 1 }        
    sphere {  m*<-4.112061469845507e-18,-5.031904368135684e-18,7.3254038446480605>, 1 }
    sphere {  m*<9.428090415820634,-1.1961587169374468e-18,-2.745929488685293>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.745929488685293>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.745929488685293>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-4.112061469845507e-18,-5.031904368135684e-18,7.3254038446480605>, <-2.2608489296671996e-18,-2.4983699987804424e-18,0.5874038446480405>, 0.5 }
    cylinder { m*<9.428090415820634,-1.1961587169374468e-18,-2.745929488685293>, <-2.2608489296671996e-18,-2.4983699987804424e-18,0.5874038446480405>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.745929488685293>, <-2.2608489296671996e-18,-2.4983699987804424e-18,0.5874038446480405>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.745929488685293>, <-2.2608489296671996e-18,-2.4983699987804424e-18,0.5874038446480405>, 0.5}

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
    sphere { m*<-2.2608489296671996e-18,-2.4983699987804424e-18,0.5874038446480405>, 1 }        
    sphere {  m*<-4.112061469845507e-18,-5.031904368135684e-18,7.3254038446480605>, 1 }
    sphere {  m*<9.428090415820634,-1.1961587169374468e-18,-2.745929488685293>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.745929488685293>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.745929488685293>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-4.112061469845507e-18,-5.031904368135684e-18,7.3254038446480605>, <-2.2608489296671996e-18,-2.4983699987804424e-18,0.5874038446480405>, 0.5 }
    cylinder { m*<9.428090415820634,-1.1961587169374468e-18,-2.745929488685293>, <-2.2608489296671996e-18,-2.4983699987804424e-18,0.5874038446480405>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.745929488685293>, <-2.2608489296671996e-18,-2.4983699987804424e-18,0.5874038446480405>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.745929488685293>, <-2.2608489296671996e-18,-2.4983699987804424e-18,0.5874038446480405>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    