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
    sphere { m*<-1.20851247995965,-0.7735192046619842,-0.8289872672098819>, 1 }        
    sphere {  m*<0.2352385414374747,-0.05969528247126274,9.040539939208221>, 1 }
    sphere {  m*<7.590589979437445,-0.14861555846561977,-5.538953350837129>, 1 }
    sphere {  m*<-5.236352098034976,4.265843711710884,-2.8919151176723807>, 1}
    sphere { m*<-2.497143353890153,-3.407099259311084,-1.4630358526324885>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2352385414374747,-0.05969528247126274,9.040539939208221>, <-1.20851247995965,-0.7735192046619842,-0.8289872672098819>, 0.5 }
    cylinder { m*<7.590589979437445,-0.14861555846561977,-5.538953350837129>, <-1.20851247995965,-0.7735192046619842,-0.8289872672098819>, 0.5}
    cylinder { m*<-5.236352098034976,4.265843711710884,-2.8919151176723807>, <-1.20851247995965,-0.7735192046619842,-0.8289872672098819>, 0.5 }
    cylinder {  m*<-2.497143353890153,-3.407099259311084,-1.4630358526324885>, <-1.20851247995965,-0.7735192046619842,-0.8289872672098819>, 0.5}

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
    sphere { m*<-1.20851247995965,-0.7735192046619842,-0.8289872672098819>, 1 }        
    sphere {  m*<0.2352385414374747,-0.05969528247126274,9.040539939208221>, 1 }
    sphere {  m*<7.590589979437445,-0.14861555846561977,-5.538953350837129>, 1 }
    sphere {  m*<-5.236352098034976,4.265843711710884,-2.8919151176723807>, 1}
    sphere { m*<-2.497143353890153,-3.407099259311084,-1.4630358526324885>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2352385414374747,-0.05969528247126274,9.040539939208221>, <-1.20851247995965,-0.7735192046619842,-0.8289872672098819>, 0.5 }
    cylinder { m*<7.590589979437445,-0.14861555846561977,-5.538953350837129>, <-1.20851247995965,-0.7735192046619842,-0.8289872672098819>, 0.5}
    cylinder { m*<-5.236352098034976,4.265843711710884,-2.8919151176723807>, <-1.20851247995965,-0.7735192046619842,-0.8289872672098819>, 0.5 }
    cylinder {  m*<-2.497143353890153,-3.407099259311084,-1.4630358526324885>, <-1.20851247995965,-0.7735192046619842,-0.8289872672098819>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    