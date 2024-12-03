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
    sphere { m*<0.36610112764382874,-5.113835215210294e-18,1.0615700973202387>, 1 }        
    sphere {  m*<0.4173631889349931,-2.77295624612255e-18,4.061133762770584>, 1 }
    sphere {  m*<8.004293712643557,2.7970685348937294e-18,-1.786799375471657>, 1 }
    sphere {  m*<-4.40421629254795,8.164965809277259,-2.1907278617210766>, 1}
    sphere { m*<-4.40421629254795,-8.164965809277259,-2.19072786172108>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4173631889349931,-2.77295624612255e-18,4.061133762770584>, <0.36610112764382874,-5.113835215210294e-18,1.0615700973202387>, 0.5 }
    cylinder { m*<8.004293712643557,2.7970685348937294e-18,-1.786799375471657>, <0.36610112764382874,-5.113835215210294e-18,1.0615700973202387>, 0.5}
    cylinder { m*<-4.40421629254795,8.164965809277259,-2.1907278617210766>, <0.36610112764382874,-5.113835215210294e-18,1.0615700973202387>, 0.5 }
    cylinder {  m*<-4.40421629254795,-8.164965809277259,-2.19072786172108>, <0.36610112764382874,-5.113835215210294e-18,1.0615700973202387>, 0.5}

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
    sphere { m*<0.36610112764382874,-5.113835215210294e-18,1.0615700973202387>, 1 }        
    sphere {  m*<0.4173631889349931,-2.77295624612255e-18,4.061133762770584>, 1 }
    sphere {  m*<8.004293712643557,2.7970685348937294e-18,-1.786799375471657>, 1 }
    sphere {  m*<-4.40421629254795,8.164965809277259,-2.1907278617210766>, 1}
    sphere { m*<-4.40421629254795,-8.164965809277259,-2.19072786172108>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4173631889349931,-2.77295624612255e-18,4.061133762770584>, <0.36610112764382874,-5.113835215210294e-18,1.0615700973202387>, 0.5 }
    cylinder { m*<8.004293712643557,2.7970685348937294e-18,-1.786799375471657>, <0.36610112764382874,-5.113835215210294e-18,1.0615700973202387>, 0.5}
    cylinder { m*<-4.40421629254795,8.164965809277259,-2.1907278617210766>, <0.36610112764382874,-5.113835215210294e-18,1.0615700973202387>, 0.5 }
    cylinder {  m*<-4.40421629254795,-8.164965809277259,-2.19072786172108>, <0.36610112764382874,-5.113835215210294e-18,1.0615700973202387>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    