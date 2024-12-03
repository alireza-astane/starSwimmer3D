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
    sphere { m*<-3.037600100457794e-18,3.48438222875015e-18,0.1673570986999538>, 1 }        
    sphere {  m*<-3.460364034535061e-18,-3.543366318611642e-18,9.257357098699952>, 1 }
    sphere {  m*<9.428090415820634,-8.825866685406213e-19,-3.1659762346333795>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-3.1659762346333795>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-3.1659762346333795>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-3.460364034535061e-18,-3.543366318611642e-18,9.257357098699952>, <-3.037600100457794e-18,3.48438222875015e-18,0.1673570986999538>, 0.5 }
    cylinder { m*<9.428090415820634,-8.825866685406213e-19,-3.1659762346333795>, <-3.037600100457794e-18,3.48438222875015e-18,0.1673570986999538>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-3.1659762346333795>, <-3.037600100457794e-18,3.48438222875015e-18,0.1673570986999538>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-3.1659762346333795>, <-3.037600100457794e-18,3.48438222875015e-18,0.1673570986999538>, 0.5}

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
    sphere { m*<-3.037600100457794e-18,3.48438222875015e-18,0.1673570986999538>, 1 }        
    sphere {  m*<-3.460364034535061e-18,-3.543366318611642e-18,9.257357098699952>, 1 }
    sphere {  m*<9.428090415820634,-8.825866685406213e-19,-3.1659762346333795>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-3.1659762346333795>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-3.1659762346333795>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-3.460364034535061e-18,-3.543366318611642e-18,9.257357098699952>, <-3.037600100457794e-18,3.48438222875015e-18,0.1673570986999538>, 0.5 }
    cylinder { m*<9.428090415820634,-8.825866685406213e-19,-3.1659762346333795>, <-3.037600100457794e-18,3.48438222875015e-18,0.1673570986999538>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-3.1659762346333795>, <-3.037600100457794e-18,3.48438222875015e-18,0.1673570986999538>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-3.1659762346333795>, <-3.037600100457794e-18,3.48438222875015e-18,0.1673570986999538>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    