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
    sphere { m*<-0.7354227169900982,-1.1221473128866934,-0.59012910964087>, 1 }        
    sphere {  m*<0.6837447772100641,-0.13220839900677595,9.259160987394282>, 1 }
    sphere {  m*<8.051531975532862,-0.41730064979903836,-5.311516441679652>, 1 }
    sphere {  m*<-6.844431218156126,6.105780723821615,-3.820709538498048>, 1}
    sphere { m*<-2.4205655758597358,-4.792059422191587,-1.3704977864949213>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6837447772100641,-0.13220839900677595,9.259160987394282>, <-0.7354227169900982,-1.1221473128866934,-0.59012910964087>, 0.5 }
    cylinder { m*<8.051531975532862,-0.41730064979903836,-5.311516441679652>, <-0.7354227169900982,-1.1221473128866934,-0.59012910964087>, 0.5}
    cylinder { m*<-6.844431218156126,6.105780723821615,-3.820709538498048>, <-0.7354227169900982,-1.1221473128866934,-0.59012910964087>, 0.5 }
    cylinder {  m*<-2.4205655758597358,-4.792059422191587,-1.3704977864949213>, <-0.7354227169900982,-1.1221473128866934,-0.59012910964087>, 0.5}

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
    sphere { m*<-0.7354227169900982,-1.1221473128866934,-0.59012910964087>, 1 }        
    sphere {  m*<0.6837447772100641,-0.13220839900677595,9.259160987394282>, 1 }
    sphere {  m*<8.051531975532862,-0.41730064979903836,-5.311516441679652>, 1 }
    sphere {  m*<-6.844431218156126,6.105780723821615,-3.820709538498048>, 1}
    sphere { m*<-2.4205655758597358,-4.792059422191587,-1.3704977864949213>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6837447772100641,-0.13220839900677595,9.259160987394282>, <-0.7354227169900982,-1.1221473128866934,-0.59012910964087>, 0.5 }
    cylinder { m*<8.051531975532862,-0.41730064979903836,-5.311516441679652>, <-0.7354227169900982,-1.1221473128866934,-0.59012910964087>, 0.5}
    cylinder { m*<-6.844431218156126,6.105780723821615,-3.820709538498048>, <-0.7354227169900982,-1.1221473128866934,-0.59012910964087>, 0.5 }
    cylinder {  m*<-2.4205655758597358,-4.792059422191587,-1.3704977864949213>, <-0.7354227169900982,-1.1221473128866934,-0.59012910964087>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    