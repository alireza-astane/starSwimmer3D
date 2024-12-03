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
    sphere { m*<0.4311890492906061,1.0229794805655017,0.12177775696762384>, 1 }        
    sphere {  m*<0.6719241540322978,1.1516895587458273,3.1093325280881743>, 1 }
    sphere {  m*<3.1658974432968625,1.1250134559518763,-1.1074317684835604>, 1 }
    sphere {  m*<-1.1904263106022834,3.3514534249841024,-0.8521680084483466>, 1}
    sphere { m*<-3.7435656548921106,-6.868795752555163,-2.297048773640425>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6719241540322978,1.1516895587458273,3.1093325280881743>, <0.4311890492906061,1.0229794805655017,0.12177775696762384>, 0.5 }
    cylinder { m*<3.1658974432968625,1.1250134559518763,-1.1074317684835604>, <0.4311890492906061,1.0229794805655017,0.12177775696762384>, 0.5}
    cylinder { m*<-1.1904263106022834,3.3514534249841024,-0.8521680084483466>, <0.4311890492906061,1.0229794805655017,0.12177775696762384>, 0.5 }
    cylinder {  m*<-3.7435656548921106,-6.868795752555163,-2.297048773640425>, <0.4311890492906061,1.0229794805655017,0.12177775696762384>, 0.5}

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
    sphere { m*<0.4311890492906061,1.0229794805655017,0.12177775696762384>, 1 }        
    sphere {  m*<0.6719241540322978,1.1516895587458273,3.1093325280881743>, 1 }
    sphere {  m*<3.1658974432968625,1.1250134559518763,-1.1074317684835604>, 1 }
    sphere {  m*<-1.1904263106022834,3.3514534249841024,-0.8521680084483466>, 1}
    sphere { m*<-3.7435656548921106,-6.868795752555163,-2.297048773640425>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6719241540322978,1.1516895587458273,3.1093325280881743>, <0.4311890492906061,1.0229794805655017,0.12177775696762384>, 0.5 }
    cylinder { m*<3.1658974432968625,1.1250134559518763,-1.1074317684835604>, <0.4311890492906061,1.0229794805655017,0.12177775696762384>, 0.5}
    cylinder { m*<-1.1904263106022834,3.3514534249841024,-0.8521680084483466>, <0.4311890492906061,1.0229794805655017,0.12177775696762384>, 0.5 }
    cylinder {  m*<-3.7435656548921106,-6.868795752555163,-2.297048773640425>, <0.4311890492906061,1.0229794805655017,0.12177775696762384>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    