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
    sphere { m*<-0.8773893883944943,-1.1889250705850685,-0.6593899523896568>, 1 }        
    sphere {  m*<0.5485826166011709,-0.3193601024250945,9.200215628475243>, 1 }
    sphere {  m*<7.903934054601145,-0.4082803784194504,-5.379277661570084>, 1 }
    sphere {  m*<-6.664342477873721,5.665872897674928,-3.620922782967837>, 1}
    sphere { m*<-2.1016690011720955,-3.8606040018379977,-1.2607391087651671>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5485826166011709,-0.3193601024250945,9.200215628475243>, <-0.8773893883944943,-1.1889250705850685,-0.6593899523896568>, 0.5 }
    cylinder { m*<7.903934054601145,-0.4082803784194504,-5.379277661570084>, <-0.8773893883944943,-1.1889250705850685,-0.6593899523896568>, 0.5}
    cylinder { m*<-6.664342477873721,5.665872897674928,-3.620922782967837>, <-0.8773893883944943,-1.1889250705850685,-0.6593899523896568>, 0.5 }
    cylinder {  m*<-2.1016690011720955,-3.8606040018379977,-1.2607391087651671>, <-0.8773893883944943,-1.1889250705850685,-0.6593899523896568>, 0.5}

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
    sphere { m*<-0.8773893883944943,-1.1889250705850685,-0.6593899523896568>, 1 }        
    sphere {  m*<0.5485826166011709,-0.3193601024250945,9.200215628475243>, 1 }
    sphere {  m*<7.903934054601145,-0.4082803784194504,-5.379277661570084>, 1 }
    sphere {  m*<-6.664342477873721,5.665872897674928,-3.620922782967837>, 1}
    sphere { m*<-2.1016690011720955,-3.8606040018379977,-1.2607391087651671>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5485826166011709,-0.3193601024250945,9.200215628475243>, <-0.8773893883944943,-1.1889250705850685,-0.6593899523896568>, 0.5 }
    cylinder { m*<7.903934054601145,-0.4082803784194504,-5.379277661570084>, <-0.8773893883944943,-1.1889250705850685,-0.6593899523896568>, 0.5}
    cylinder { m*<-6.664342477873721,5.665872897674928,-3.620922782967837>, <-0.8773893883944943,-1.1889250705850685,-0.6593899523896568>, 0.5 }
    cylinder {  m*<-2.1016690011720955,-3.8606040018379977,-1.2607391087651671>, <-0.8773893883944943,-1.1889250705850685,-0.6593899523896568>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    