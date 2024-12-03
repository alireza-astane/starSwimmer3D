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
    sphere { m*<-0.15826578046295742,-0.07983529184621549,-0.3198003370162923>, 1 }        
    sphere {  m*<0.13527055725207832,0.07710519681499431,3.3230247805700306>, 1 }
    sphere {  m*<2.5764426135432994,0.022198683540158623,-1.5490098624674757>, 1 }
    sphere {  m*<-1.7798811403558477,2.2486386525723834,-1.2937461024322625>, 1}
    sphere { m*<-1.5120939193180158,-2.639053289831514,-1.1041998172696899>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.13527055725207832,0.07710519681499431,3.3230247805700306>, <-0.15826578046295742,-0.07983529184621549,-0.3198003370162923>, 0.5 }
    cylinder { m*<2.5764426135432994,0.022198683540158623,-1.5490098624674757>, <-0.15826578046295742,-0.07983529184621549,-0.3198003370162923>, 0.5}
    cylinder { m*<-1.7798811403558477,2.2486386525723834,-1.2937461024322625>, <-0.15826578046295742,-0.07983529184621549,-0.3198003370162923>, 0.5 }
    cylinder {  m*<-1.5120939193180158,-2.639053289831514,-1.1041998172696899>, <-0.15826578046295742,-0.07983529184621549,-0.3198003370162923>, 0.5}

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
    sphere { m*<-0.15826578046295742,-0.07983529184621549,-0.3198003370162923>, 1 }        
    sphere {  m*<0.13527055725207832,0.07710519681499431,3.3230247805700306>, 1 }
    sphere {  m*<2.5764426135432994,0.022198683540158623,-1.5490098624674757>, 1 }
    sphere {  m*<-1.7798811403558477,2.2486386525723834,-1.2937461024322625>, 1}
    sphere { m*<-1.5120939193180158,-2.639053289831514,-1.1041998172696899>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.13527055725207832,0.07710519681499431,3.3230247805700306>, <-0.15826578046295742,-0.07983529184621549,-0.3198003370162923>, 0.5 }
    cylinder { m*<2.5764426135432994,0.022198683540158623,-1.5490098624674757>, <-0.15826578046295742,-0.07983529184621549,-0.3198003370162923>, 0.5}
    cylinder { m*<-1.7798811403558477,2.2486386525723834,-1.2937461024322625>, <-0.15826578046295742,-0.07983529184621549,-0.3198003370162923>, 0.5 }
    cylinder {  m*<-1.5120939193180158,-2.639053289831514,-1.1041998172696899>, <-0.15826578046295742,-0.07983529184621549,-0.3198003370162923>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    