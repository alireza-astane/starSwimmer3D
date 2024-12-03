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
    sphere { m*<-2.809778756099888e-18,2.5801156749085175e-18,0.09812725244126003>, 1 }        
    sphere {  m*<-4.625931147520304e-18,5.484665705954686e-19,9.566127252441254>, 1 }
    sphere {  m*<9.428090415820634,1.393423743571582e-19,-3.2352060808920746>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-3.2352060808920746>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-3.2352060808920746>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-4.625931147520304e-18,5.484665705954686e-19,9.566127252441254>, <-2.809778756099888e-18,2.5801156749085175e-18,0.09812725244126003>, 0.5 }
    cylinder { m*<9.428090415820634,1.393423743571582e-19,-3.2352060808920746>, <-2.809778756099888e-18,2.5801156749085175e-18,0.09812725244126003>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-3.2352060808920746>, <-2.809778756099888e-18,2.5801156749085175e-18,0.09812725244126003>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-3.2352060808920746>, <-2.809778756099888e-18,2.5801156749085175e-18,0.09812725244126003>, 0.5}

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
    sphere { m*<-2.809778756099888e-18,2.5801156749085175e-18,0.09812725244126003>, 1 }        
    sphere {  m*<-4.625931147520304e-18,5.484665705954686e-19,9.566127252441254>, 1 }
    sphere {  m*<9.428090415820634,1.393423743571582e-19,-3.2352060808920746>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-3.2352060808920746>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-3.2352060808920746>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-4.625931147520304e-18,5.484665705954686e-19,9.566127252441254>, <-2.809778756099888e-18,2.5801156749085175e-18,0.09812725244126003>, 0.5 }
    cylinder { m*<9.428090415820634,1.393423743571582e-19,-3.2352060808920746>, <-2.809778756099888e-18,2.5801156749085175e-18,0.09812725244126003>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-3.2352060808920746>, <-2.809778756099888e-18,2.5801156749085175e-18,0.09812725244126003>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-3.2352060808920746>, <-2.809778756099888e-18,2.5801156749085175e-18,0.09812725244126003>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    