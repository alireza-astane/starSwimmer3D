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
    sphere { m*<-0.16283084545679172,-0.08227602384035128,-0.37645340270622984>, 1 }        
    sphere {  m*<0.15710610874491615,0.08877967006130083,3.59400688811298>, 1 }
    sphere {  m*<2.571877548549465,0.019757951546022848,-1.605662928157413>, 1 }
    sphere {  m*<-1.784446205349682,2.246197920578248,-1.3503991681221998>, 1}
    sphere { m*<-1.51665898431185,-2.6414940218256495,-1.1608528829596272>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.15710610874491615,0.08877967006130083,3.59400688811298>, <-0.16283084545679172,-0.08227602384035128,-0.37645340270622984>, 0.5 }
    cylinder { m*<2.571877548549465,0.019757951546022848,-1.605662928157413>, <-0.16283084545679172,-0.08227602384035128,-0.37645340270622984>, 0.5}
    cylinder { m*<-1.784446205349682,2.246197920578248,-1.3503991681221998>, <-0.16283084545679172,-0.08227602384035128,-0.37645340270622984>, 0.5 }
    cylinder {  m*<-1.51665898431185,-2.6414940218256495,-1.1608528829596272>, <-0.16283084545679172,-0.08227602384035128,-0.37645340270622984>, 0.5}

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
    sphere { m*<-0.16283084545679172,-0.08227602384035128,-0.37645340270622984>, 1 }        
    sphere {  m*<0.15710610874491615,0.08877967006130083,3.59400688811298>, 1 }
    sphere {  m*<2.571877548549465,0.019757951546022848,-1.605662928157413>, 1 }
    sphere {  m*<-1.784446205349682,2.246197920578248,-1.3503991681221998>, 1}
    sphere { m*<-1.51665898431185,-2.6414940218256495,-1.1608528829596272>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.15710610874491615,0.08877967006130083,3.59400688811298>, <-0.16283084545679172,-0.08227602384035128,-0.37645340270622984>, 0.5 }
    cylinder { m*<2.571877548549465,0.019757951546022848,-1.605662928157413>, <-0.16283084545679172,-0.08227602384035128,-0.37645340270622984>, 0.5}
    cylinder { m*<-1.784446205349682,2.246197920578248,-1.3503991681221998>, <-0.16283084545679172,-0.08227602384035128,-0.37645340270622984>, 0.5 }
    cylinder {  m*<-1.51665898431185,-2.6414940218256495,-1.1608528829596272>, <-0.16283084545679172,-0.08227602384035128,-0.37645340270622984>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    