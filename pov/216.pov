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
    sphere { m*<-3.0605760294165757e-18,1.6025176569119476e-18,0.2779753557484308>, 1 }        
    sphere {  m*<-6.606832509712794e-18,-2.337293251656187e-18,8.758975355748442>, 1 }
    sphere {  m*<9.428090415820634,-1.8300587280022178e-18,-3.055357977584902>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-3.055357977584902>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-3.055357977584902>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-6.606832509712794e-18,-2.337293251656187e-18,8.758975355748442>, <-3.0605760294165757e-18,1.6025176569119476e-18,0.2779753557484308>, 0.5 }
    cylinder { m*<9.428090415820634,-1.8300587280022178e-18,-3.055357977584902>, <-3.0605760294165757e-18,1.6025176569119476e-18,0.2779753557484308>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-3.055357977584902>, <-3.0605760294165757e-18,1.6025176569119476e-18,0.2779753557484308>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-3.055357977584902>, <-3.0605760294165757e-18,1.6025176569119476e-18,0.2779753557484308>, 0.5}

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
    sphere { m*<-3.0605760294165757e-18,1.6025176569119476e-18,0.2779753557484308>, 1 }        
    sphere {  m*<-6.606832509712794e-18,-2.337293251656187e-18,8.758975355748442>, 1 }
    sphere {  m*<9.428090415820634,-1.8300587280022178e-18,-3.055357977584902>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-3.055357977584902>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-3.055357977584902>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-6.606832509712794e-18,-2.337293251656187e-18,8.758975355748442>, <-3.0605760294165757e-18,1.6025176569119476e-18,0.2779753557484308>, 0.5 }
    cylinder { m*<9.428090415820634,-1.8300587280022178e-18,-3.055357977584902>, <-3.0605760294165757e-18,1.6025176569119476e-18,0.2779753557484308>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-3.055357977584902>, <-3.0605760294165757e-18,1.6025176569119476e-18,0.2779753557484308>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-3.055357977584902>, <-3.0605760294165757e-18,1.6025176569119476e-18,0.2779753557484308>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    