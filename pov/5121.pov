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
    sphere { m*<-0.40774847734097763,-0.1437247222243361,-1.618942273662887>, 1 }        
    sphere {  m*<0.47679174358922877,0.28952245489081646,8.33242997501362>, 1 }
    sphere {  m*<3.092393353244964,-0.014475183147203224,-3.226936072324704>, 1 }
    sphere {  m*<-2.036588953246107,2.184830935972223,-2.580557341163811>, 1}
    sphere { m*<-1.7688017322082754,-2.7028610064316743,-2.3910110560012408>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.47679174358922877,0.28952245489081646,8.33242997501362>, <-0.40774847734097763,-0.1437247222243361,-1.618942273662887>, 0.5 }
    cylinder { m*<3.092393353244964,-0.014475183147203224,-3.226936072324704>, <-0.40774847734097763,-0.1437247222243361,-1.618942273662887>, 0.5}
    cylinder { m*<-2.036588953246107,2.184830935972223,-2.580557341163811>, <-0.40774847734097763,-0.1437247222243361,-1.618942273662887>, 0.5 }
    cylinder {  m*<-1.7688017322082754,-2.7028610064316743,-2.3910110560012408>, <-0.40774847734097763,-0.1437247222243361,-1.618942273662887>, 0.5}

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
    sphere { m*<-0.40774847734097763,-0.1437247222243361,-1.618942273662887>, 1 }        
    sphere {  m*<0.47679174358922877,0.28952245489081646,8.33242997501362>, 1 }
    sphere {  m*<3.092393353244964,-0.014475183147203224,-3.226936072324704>, 1 }
    sphere {  m*<-2.036588953246107,2.184830935972223,-2.580557341163811>, 1}
    sphere { m*<-1.7688017322082754,-2.7028610064316743,-2.3910110560012408>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.47679174358922877,0.28952245489081646,8.33242997501362>, <-0.40774847734097763,-0.1437247222243361,-1.618942273662887>, 0.5 }
    cylinder { m*<3.092393353244964,-0.014475183147203224,-3.226936072324704>, <-0.40774847734097763,-0.1437247222243361,-1.618942273662887>, 0.5}
    cylinder { m*<-2.036588953246107,2.184830935972223,-2.580557341163811>, <-0.40774847734097763,-0.1437247222243361,-1.618942273662887>, 0.5 }
    cylinder {  m*<-1.7688017322082754,-2.7028610064316743,-2.3910110560012408>, <-0.40774847734097763,-0.1437247222243361,-1.618942273662887>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    