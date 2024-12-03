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
    sphere { m*<0.3547786401965346,0.8785365571436661,0.077506048130173>, 1 }        
    sphere {  m*<0.5955137449382264,1.0072466353239917,3.0650608192507263>, 1 }
    sphere {  m*<3.089487034202792,0.9805705325300407,-1.1517034773210106>, 1 }
    sphere {  m*<-1.266836719696355,3.2070105015622694,-0.8964397172857966>, 1}
    sphere { m*<-3.494606034614682,-6.398173283794529,-2.1528031407236448>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5955137449382264,1.0072466353239917,3.0650608192507263>, <0.3547786401965346,0.8785365571436661,0.077506048130173>, 0.5 }
    cylinder { m*<3.089487034202792,0.9805705325300407,-1.1517034773210106>, <0.3547786401965346,0.8785365571436661,0.077506048130173>, 0.5}
    cylinder { m*<-1.266836719696355,3.2070105015622694,-0.8964397172857966>, <0.3547786401965346,0.8785365571436661,0.077506048130173>, 0.5 }
    cylinder {  m*<-3.494606034614682,-6.398173283794529,-2.1528031407236448>, <0.3547786401965346,0.8785365571436661,0.077506048130173>, 0.5}

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
    sphere { m*<0.3547786401965346,0.8785365571436661,0.077506048130173>, 1 }        
    sphere {  m*<0.5955137449382264,1.0072466353239917,3.0650608192507263>, 1 }
    sphere {  m*<3.089487034202792,0.9805705325300407,-1.1517034773210106>, 1 }
    sphere {  m*<-1.266836719696355,3.2070105015622694,-0.8964397172857966>, 1}
    sphere { m*<-3.494606034614682,-6.398173283794529,-2.1528031407236448>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5955137449382264,1.0072466353239917,3.0650608192507263>, <0.3547786401965346,0.8785365571436661,0.077506048130173>, 0.5 }
    cylinder { m*<3.089487034202792,0.9805705325300407,-1.1517034773210106>, <0.3547786401965346,0.8785365571436661,0.077506048130173>, 0.5}
    cylinder { m*<-1.266836719696355,3.2070105015622694,-0.8964397172857966>, <0.3547786401965346,0.8785365571436661,0.077506048130173>, 0.5 }
    cylinder {  m*<-3.494606034614682,-6.398173283794529,-2.1528031407236448>, <0.3547786401965346,0.8785365571436661,0.077506048130173>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    