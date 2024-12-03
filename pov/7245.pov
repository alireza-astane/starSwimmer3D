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
    sphere { m*<-0.6953563229761923,-1.0348905300009703,-0.5715748614248357>, 1 }        
    sphere {  m*<0.7238111712239702,-0.044951616121052584,9.277715235610318>, 1 }
    sphere {  m*<8.091598369546766,-0.33004386691331566,-5.292962193463617>, 1 }
    sphere {  m*<-6.804364824142218,6.193037506707335,-3.8021552902820126>, 1}
    sphere { m*<-2.630013405296257,-5.248195898378726,-1.4674904685497632>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7238111712239702,-0.044951616121052584,9.277715235610318>, <-0.6953563229761923,-1.0348905300009703,-0.5715748614248357>, 0.5 }
    cylinder { m*<8.091598369546766,-0.33004386691331566,-5.292962193463617>, <-0.6953563229761923,-1.0348905300009703,-0.5715748614248357>, 0.5}
    cylinder { m*<-6.804364824142218,6.193037506707335,-3.8021552902820126>, <-0.6953563229761923,-1.0348905300009703,-0.5715748614248357>, 0.5 }
    cylinder {  m*<-2.630013405296257,-5.248195898378726,-1.4674904685497632>, <-0.6953563229761923,-1.0348905300009703,-0.5715748614248357>, 0.5}

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
    sphere { m*<-0.6953563229761923,-1.0348905300009703,-0.5715748614248357>, 1 }        
    sphere {  m*<0.7238111712239702,-0.044951616121052584,9.277715235610318>, 1 }
    sphere {  m*<8.091598369546766,-0.33004386691331566,-5.292962193463617>, 1 }
    sphere {  m*<-6.804364824142218,6.193037506707335,-3.8021552902820126>, 1}
    sphere { m*<-2.630013405296257,-5.248195898378726,-1.4674904685497632>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7238111712239702,-0.044951616121052584,9.277715235610318>, <-0.6953563229761923,-1.0348905300009703,-0.5715748614248357>, 0.5 }
    cylinder { m*<8.091598369546766,-0.33004386691331566,-5.292962193463617>, <-0.6953563229761923,-1.0348905300009703,-0.5715748614248357>, 0.5}
    cylinder { m*<-6.804364824142218,6.193037506707335,-3.8021552902820126>, <-0.6953563229761923,-1.0348905300009703,-0.5715748614248357>, 0.5 }
    cylinder {  m*<-2.630013405296257,-5.248195898378726,-1.4674904685497632>, <-0.6953563229761923,-1.0348905300009703,-0.5715748614248357>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    