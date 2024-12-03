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
    sphere { m*<-0.42078618099046294,-0.4369303727221363,-0.44442484724220543>, 1 }        
    sphere {  m*<0.9983813132096985,0.553008541157781,9.404865249792941>, 1 }
    sphere {  m*<8.366168511532495,0.2679162903655188,-5.16581217928099>, 1 }
    sphere {  m*<-6.5297946821565,6.790997663986155,-3.675005276099382>, 1}
    sphere { m*<-3.941435970989682,-8.104218180953827,-2.0747939306866146>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9983813132096985,0.553008541157781,9.404865249792941>, <-0.42078618099046294,-0.4369303727221363,-0.44442484724220543>, 0.5 }
    cylinder { m*<8.366168511532495,0.2679162903655188,-5.16581217928099>, <-0.42078618099046294,-0.4369303727221363,-0.44442484724220543>, 0.5}
    cylinder { m*<-6.5297946821565,6.790997663986155,-3.675005276099382>, <-0.42078618099046294,-0.4369303727221363,-0.44442484724220543>, 0.5 }
    cylinder {  m*<-3.941435970989682,-8.104218180953827,-2.0747939306866146>, <-0.42078618099046294,-0.4369303727221363,-0.44442484724220543>, 0.5}

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
    sphere { m*<-0.42078618099046294,-0.4369303727221363,-0.44442484724220543>, 1 }        
    sphere {  m*<0.9983813132096985,0.553008541157781,9.404865249792941>, 1 }
    sphere {  m*<8.366168511532495,0.2679162903655188,-5.16581217928099>, 1 }
    sphere {  m*<-6.5297946821565,6.790997663986155,-3.675005276099382>, 1}
    sphere { m*<-3.941435970989682,-8.104218180953827,-2.0747939306866146>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9983813132096985,0.553008541157781,9.404865249792941>, <-0.42078618099046294,-0.4369303727221363,-0.44442484724220543>, 0.5 }
    cylinder { m*<8.366168511532495,0.2679162903655188,-5.16581217928099>, <-0.42078618099046294,-0.4369303727221363,-0.44442484724220543>, 0.5}
    cylinder { m*<-6.5297946821565,6.790997663986155,-3.675005276099382>, <-0.42078618099046294,-0.4369303727221363,-0.44442484724220543>, 0.5 }
    cylinder {  m*<-3.941435970989682,-8.104218180953827,-2.0747939306866146>, <-0.42078618099046294,-0.4369303727221363,-0.44442484724220543>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    