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
    sphere { m*<-1.0133266208672187e-19,2.1749922724156882e-18,0.034950977918466714>, 1 }        
    sphere {  m*<-1.1448449460585286e-18,4.5513086702524375e-19,9.845950977918463>, 1 }
    sphere {  m*<9.428090415820634,5.564476818741918e-19,-3.2983823554148675>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-3.2983823554148675>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-3.2983823554148675>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-1.1448449460585286e-18,4.5513086702524375e-19,9.845950977918463>, <-1.0133266208672187e-19,2.1749922724156882e-18,0.034950977918466714>, 0.5 }
    cylinder { m*<9.428090415820634,5.564476818741918e-19,-3.2983823554148675>, <-1.0133266208672187e-19,2.1749922724156882e-18,0.034950977918466714>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-3.2983823554148675>, <-1.0133266208672187e-19,2.1749922724156882e-18,0.034950977918466714>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-3.2983823554148675>, <-1.0133266208672187e-19,2.1749922724156882e-18,0.034950977918466714>, 0.5}

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
    sphere { m*<-1.0133266208672187e-19,2.1749922724156882e-18,0.034950977918466714>, 1 }        
    sphere {  m*<-1.1448449460585286e-18,4.5513086702524375e-19,9.845950977918463>, 1 }
    sphere {  m*<9.428090415820634,5.564476818741918e-19,-3.2983823554148675>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-3.2983823554148675>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-3.2983823554148675>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-1.1448449460585286e-18,4.5513086702524375e-19,9.845950977918463>, <-1.0133266208672187e-19,2.1749922724156882e-18,0.034950977918466714>, 0.5 }
    cylinder { m*<9.428090415820634,5.564476818741918e-19,-3.2983823554148675>, <-1.0133266208672187e-19,2.1749922724156882e-18,0.034950977918466714>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-3.2983823554148675>, <-1.0133266208672187e-19,2.1749922724156882e-18,0.034950977918466714>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-3.2983823554148675>, <-1.0133266208672187e-19,2.1749922724156882e-18,0.034950977918466714>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    