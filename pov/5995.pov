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
    sphere { m*<-1.5855007804170718,-0.18531988575899577,-1.0288586973185398>, 1 }        
    sphere {  m*<-0.12178983170873733,0.2769004200312004,8.852589190251825>, 1 }
    sphere {  m*<7.214206508212933,0.11491590621507758,-5.710445879488132>, 1 }
    sphere {  m*<-3.262557947187345,2.1440384552639613,-1.9015304140688665>, 1}
    sphere { m*<-2.994770726149514,-2.743653487139936,-1.711984128906296>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.12178983170873733,0.2769004200312004,8.852589190251825>, <-1.5855007804170718,-0.18531988575899577,-1.0288586973185398>, 0.5 }
    cylinder { m*<7.214206508212933,0.11491590621507758,-5.710445879488132>, <-1.5855007804170718,-0.18531988575899577,-1.0288586973185398>, 0.5}
    cylinder { m*<-3.262557947187345,2.1440384552639613,-1.9015304140688665>, <-1.5855007804170718,-0.18531988575899577,-1.0288586973185398>, 0.5 }
    cylinder {  m*<-2.994770726149514,-2.743653487139936,-1.711984128906296>, <-1.5855007804170718,-0.18531988575899577,-1.0288586973185398>, 0.5}

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
    sphere { m*<-1.5855007804170718,-0.18531988575899577,-1.0288586973185398>, 1 }        
    sphere {  m*<-0.12178983170873733,0.2769004200312004,8.852589190251825>, 1 }
    sphere {  m*<7.214206508212933,0.11491590621507758,-5.710445879488132>, 1 }
    sphere {  m*<-3.262557947187345,2.1440384552639613,-1.9015304140688665>, 1}
    sphere { m*<-2.994770726149514,-2.743653487139936,-1.711984128906296>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.12178983170873733,0.2769004200312004,8.852589190251825>, <-1.5855007804170718,-0.18531988575899577,-1.0288586973185398>, 0.5 }
    cylinder { m*<7.214206508212933,0.11491590621507758,-5.710445879488132>, <-1.5855007804170718,-0.18531988575899577,-1.0288586973185398>, 0.5}
    cylinder { m*<-3.262557947187345,2.1440384552639613,-1.9015304140688665>, <-1.5855007804170718,-0.18531988575899577,-1.0288586973185398>, 0.5 }
    cylinder {  m*<-2.994770726149514,-2.743653487139936,-1.711984128906296>, <-1.5855007804170718,-0.18531988575899577,-1.0288586973185398>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    