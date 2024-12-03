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
    sphere { m*<-0.25347171670128016,-0.13073756266215678,-1.5013187090584428>, 1 }        
    sphere {  m*<0.4905602470203736,0.26706238563383455,7.732217236884796>, 1 }
    sphere {  m*<2.481236677304977,-0.028703587275782573,-2.7305282345096242>, 1 }
    sphere {  m*<-1.87508707659417,2.1977363817564424,-2.4752644744744106>, 1}
    sphere { m*<-1.6072998555563383,-2.689955560647455,-2.285718189311838>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4905602470203736,0.26706238563383455,7.732217236884796>, <-0.25347171670128016,-0.13073756266215678,-1.5013187090584428>, 0.5 }
    cylinder { m*<2.481236677304977,-0.028703587275782573,-2.7305282345096242>, <-0.25347171670128016,-0.13073756266215678,-1.5013187090584428>, 0.5}
    cylinder { m*<-1.87508707659417,2.1977363817564424,-2.4752644744744106>, <-0.25347171670128016,-0.13073756266215678,-1.5013187090584428>, 0.5 }
    cylinder {  m*<-1.6072998555563383,-2.689955560647455,-2.285718189311838>, <-0.25347171670128016,-0.13073756266215678,-1.5013187090584428>, 0.5}

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
    sphere { m*<-0.25347171670128016,-0.13073756266215678,-1.5013187090584428>, 1 }        
    sphere {  m*<0.4905602470203736,0.26706238563383455,7.732217236884796>, 1 }
    sphere {  m*<2.481236677304977,-0.028703587275782573,-2.7305282345096242>, 1 }
    sphere {  m*<-1.87508707659417,2.1977363817564424,-2.4752644744744106>, 1}
    sphere { m*<-1.6072998555563383,-2.689955560647455,-2.285718189311838>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4905602470203736,0.26706238563383455,7.732217236884796>, <-0.25347171670128016,-0.13073756266215678,-1.5013187090584428>, 0.5 }
    cylinder { m*<2.481236677304977,-0.028703587275782573,-2.7305282345096242>, <-0.25347171670128016,-0.13073756266215678,-1.5013187090584428>, 0.5}
    cylinder { m*<-1.87508707659417,2.1977363817564424,-2.4752644744744106>, <-0.25347171670128016,-0.13073756266215678,-1.5013187090584428>, 0.5 }
    cylinder {  m*<-1.6072998555563383,-2.689955560647455,-2.285718189311838>, <-0.25347171670128016,-0.13073756266215678,-1.5013187090584428>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    