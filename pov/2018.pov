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
    sphere { m*<1.264726646491266,0.025746693479111335,0.6136585101372065>, 1 }        
    sphere {  m*<1.5089734603506373,0.0276062974489364,3.6036985672877186>, 1 }
    sphere {  m*<4.002220649413174,0.027606297448936407,-0.6135836412028979>, 1 }
    sphere {  m*<-3.6464159331267134,8.075984370742182,-2.290138386998497>, 1}
    sphere { m*<-3.697073508911592,-8.145404078703658,-2.3193996804433876>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.5089734603506373,0.0276062974489364,3.6036985672877186>, <1.264726646491266,0.025746693479111335,0.6136585101372065>, 0.5 }
    cylinder { m*<4.002220649413174,0.027606297448936407,-0.6135836412028979>, <1.264726646491266,0.025746693479111335,0.6136585101372065>, 0.5}
    cylinder { m*<-3.6464159331267134,8.075984370742182,-2.290138386998497>, <1.264726646491266,0.025746693479111335,0.6136585101372065>, 0.5 }
    cylinder {  m*<-3.697073508911592,-8.145404078703658,-2.3193996804433876>, <1.264726646491266,0.025746693479111335,0.6136585101372065>, 0.5}

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
    sphere { m*<1.264726646491266,0.025746693479111335,0.6136585101372065>, 1 }        
    sphere {  m*<1.5089734603506373,0.0276062974489364,3.6036985672877186>, 1 }
    sphere {  m*<4.002220649413174,0.027606297448936407,-0.6135836412028979>, 1 }
    sphere {  m*<-3.6464159331267134,8.075984370742182,-2.290138386998497>, 1}
    sphere { m*<-3.697073508911592,-8.145404078703658,-2.3193996804433876>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.5089734603506373,0.0276062974489364,3.6036985672877186>, <1.264726646491266,0.025746693479111335,0.6136585101372065>, 0.5 }
    cylinder { m*<4.002220649413174,0.027606297448936407,-0.6135836412028979>, <1.264726646491266,0.025746693479111335,0.6136585101372065>, 0.5}
    cylinder { m*<-3.6464159331267134,8.075984370742182,-2.290138386998497>, <1.264726646491266,0.025746693479111335,0.6136585101372065>, 0.5 }
    cylinder {  m*<-3.697073508911592,-8.145404078703658,-2.3193996804433876>, <1.264726646491266,0.025746693479111335,0.6136585101372065>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    