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
    sphere { m*<1.1807363318886595,0.17187385425614815,0.5639974445415122>, 1 }        
    sphere {  m*<1.4249427678015596,0.184752647294714,3.5540131634331313>, 1 }
    sphere {  m*<3.9181899568640968,0.184752647294714,-0.663269045057487>, 1 }
    sphere {  m*<-3.3912483058345635,7.565934741005794,-2.139262281779833>, 1}
    sphere { m*<-3.7362173091193545,-8.03578805569924,-2.3425460465512167>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.4249427678015596,0.184752647294714,3.5540131634331313>, <1.1807363318886595,0.17187385425614815,0.5639974445415122>, 0.5 }
    cylinder { m*<3.9181899568640968,0.184752647294714,-0.663269045057487>, <1.1807363318886595,0.17187385425614815,0.5639974445415122>, 0.5}
    cylinder { m*<-3.3912483058345635,7.565934741005794,-2.139262281779833>, <1.1807363318886595,0.17187385425614815,0.5639974445415122>, 0.5 }
    cylinder {  m*<-3.7362173091193545,-8.03578805569924,-2.3425460465512167>, <1.1807363318886595,0.17187385425614815,0.5639974445415122>, 0.5}

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
    sphere { m*<1.1807363318886595,0.17187385425614815,0.5639974445415122>, 1 }        
    sphere {  m*<1.4249427678015596,0.184752647294714,3.5540131634331313>, 1 }
    sphere {  m*<3.9181899568640968,0.184752647294714,-0.663269045057487>, 1 }
    sphere {  m*<-3.3912483058345635,7.565934741005794,-2.139262281779833>, 1}
    sphere { m*<-3.7362173091193545,-8.03578805569924,-2.3425460465512167>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.4249427678015596,0.184752647294714,3.5540131634331313>, <1.1807363318886595,0.17187385425614815,0.5639974445415122>, 0.5 }
    cylinder { m*<3.9181899568640968,0.184752647294714,-0.663269045057487>, <1.1807363318886595,0.17187385425614815,0.5639974445415122>, 0.5}
    cylinder { m*<-3.3912483058345635,7.565934741005794,-2.139262281779833>, <1.1807363318886595,0.17187385425614815,0.5639974445415122>, 0.5 }
    cylinder {  m*<-3.7362173091193545,-8.03578805569924,-2.3425460465512167>, <1.1807363318886595,0.17187385425614815,0.5639974445415122>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    