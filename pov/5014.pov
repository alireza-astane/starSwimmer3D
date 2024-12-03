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
    sphere { m*<-0.2839019990625417,-0.13885641514696992,-1.66820975090698>, 1 }        
    sphere {  m*<0.5282455943874689,0.29067630220131185,8.289497815248692>, 1 }
    sphere {  m*<2.5452901617681714,-0.03342495872348544,-2.943192176441577>, 1 }
    sphere {  m*<-1.906375881130625,2.1896267703889447,-2.6407024387011506>, 1}
    sphere { m*<-1.6385886600927932,-2.6980651720149527,-2.45115615353858>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5282455943874689,0.29067630220131185,8.289497815248692>, <-0.2839019990625417,-0.13885641514696992,-1.66820975090698>, 0.5 }
    cylinder { m*<2.5452901617681714,-0.03342495872348544,-2.943192176441577>, <-0.2839019990625417,-0.13885641514696992,-1.66820975090698>, 0.5}
    cylinder { m*<-1.906375881130625,2.1896267703889447,-2.6407024387011506>, <-0.2839019990625417,-0.13885641514696992,-1.66820975090698>, 0.5 }
    cylinder {  m*<-1.6385886600927932,-2.6980651720149527,-2.45115615353858>, <-0.2839019990625417,-0.13885641514696992,-1.66820975090698>, 0.5}

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
    sphere { m*<-0.2839019990625417,-0.13885641514696992,-1.66820975090698>, 1 }        
    sphere {  m*<0.5282455943874689,0.29067630220131185,8.289497815248692>, 1 }
    sphere {  m*<2.5452901617681714,-0.03342495872348544,-2.943192176441577>, 1 }
    sphere {  m*<-1.906375881130625,2.1896267703889447,-2.6407024387011506>, 1}
    sphere { m*<-1.6385886600927932,-2.6980651720149527,-2.45115615353858>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5282455943874689,0.29067630220131185,8.289497815248692>, <-0.2839019990625417,-0.13885641514696992,-1.66820975090698>, 0.5 }
    cylinder { m*<2.5452901617681714,-0.03342495872348544,-2.943192176441577>, <-0.2839019990625417,-0.13885641514696992,-1.66820975090698>, 0.5}
    cylinder { m*<-1.906375881130625,2.1896267703889447,-2.6407024387011506>, <-0.2839019990625417,-0.13885641514696992,-1.66820975090698>, 0.5 }
    cylinder {  m*<-1.6385886600927932,-2.6980651720149527,-2.45115615353858>, <-0.2839019990625417,-0.13885641514696992,-1.66820975090698>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    