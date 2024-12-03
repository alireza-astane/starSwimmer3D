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
    sphere { m*<0.01407510820364559,-5.691646348715918e-18,1.1908006607023778>, 1 }        
    sphere {  m*<0.01585194679923428,-5.307873936873998e-18,4.190800187129242>, 1 }
    sphere {  m*<9.374205589322488,2.3652225895937504e-19,-2.124747689462314>, 1 }
    sphere {  m*<-4.7019436206644905,8.164965809277259,-2.139740113949661>, 1}
    sphere { m*<-4.7019436206644905,-8.164965809277259,-2.1397401139496637>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.01585194679923428,-5.307873936873998e-18,4.190800187129242>, <0.01407510820364559,-5.691646348715918e-18,1.1908006607023778>, 0.5 }
    cylinder { m*<9.374205589322488,2.3652225895937504e-19,-2.124747689462314>, <0.01407510820364559,-5.691646348715918e-18,1.1908006607023778>, 0.5}
    cylinder { m*<-4.7019436206644905,8.164965809277259,-2.139740113949661>, <0.01407510820364559,-5.691646348715918e-18,1.1908006607023778>, 0.5 }
    cylinder {  m*<-4.7019436206644905,-8.164965809277259,-2.1397401139496637>, <0.01407510820364559,-5.691646348715918e-18,1.1908006607023778>, 0.5}

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
    sphere { m*<0.01407510820364559,-5.691646348715918e-18,1.1908006607023778>, 1 }        
    sphere {  m*<0.01585194679923428,-5.307873936873998e-18,4.190800187129242>, 1 }
    sphere {  m*<9.374205589322488,2.3652225895937504e-19,-2.124747689462314>, 1 }
    sphere {  m*<-4.7019436206644905,8.164965809277259,-2.139740113949661>, 1}
    sphere { m*<-4.7019436206644905,-8.164965809277259,-2.1397401139496637>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.01585194679923428,-5.307873936873998e-18,4.190800187129242>, <0.01407510820364559,-5.691646348715918e-18,1.1908006607023778>, 0.5 }
    cylinder { m*<9.374205589322488,2.3652225895937504e-19,-2.124747689462314>, <0.01407510820364559,-5.691646348715918e-18,1.1908006607023778>, 0.5}
    cylinder { m*<-4.7019436206644905,8.164965809277259,-2.139740113949661>, <0.01407510820364559,-5.691646348715918e-18,1.1908006607023778>, 0.5 }
    cylinder {  m*<-4.7019436206644905,-8.164965809277259,-2.1397401139496637>, <0.01407510820364559,-5.691646348715918e-18,1.1908006607023778>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    