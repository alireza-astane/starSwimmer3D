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
    sphere { m*<-1.370464507506571,-0.5451310134559277,-0.9120912508546379>, 1 }        
    sphere {  m*<0.0828371626054687,0.06687352133421426,8.962876678390353>, 1 }
    sphere {  m*<7.43818860060544,-0.022046754660142998,-5.616616611655001>, 1 }
    sphere {  m*<-4.473284240111975,3.4785127368982143,-2.502119472324625>, 1}
    sphere { m*<-2.7005423621448097,-3.152818103165367,-1.567208968729166>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.0828371626054687,0.06687352133421426,8.962876678390353>, <-1.370464507506571,-0.5451310134559277,-0.9120912508546379>, 0.5 }
    cylinder { m*<7.43818860060544,-0.022046754660142998,-5.616616611655001>, <-1.370464507506571,-0.5451310134559277,-0.9120912508546379>, 0.5}
    cylinder { m*<-4.473284240111975,3.4785127368982143,-2.502119472324625>, <-1.370464507506571,-0.5451310134559277,-0.9120912508546379>, 0.5 }
    cylinder {  m*<-2.7005423621448097,-3.152818103165367,-1.567208968729166>, <-1.370464507506571,-0.5451310134559277,-0.9120912508546379>, 0.5}

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
    sphere { m*<-1.370464507506571,-0.5451310134559277,-0.9120912508546379>, 1 }        
    sphere {  m*<0.0828371626054687,0.06687352133421426,8.962876678390353>, 1 }
    sphere {  m*<7.43818860060544,-0.022046754660142998,-5.616616611655001>, 1 }
    sphere {  m*<-4.473284240111975,3.4785127368982143,-2.502119472324625>, 1}
    sphere { m*<-2.7005423621448097,-3.152818103165367,-1.567208968729166>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.0828371626054687,0.06687352133421426,8.962876678390353>, <-1.370464507506571,-0.5451310134559277,-0.9120912508546379>, 0.5 }
    cylinder { m*<7.43818860060544,-0.022046754660142998,-5.616616611655001>, <-1.370464507506571,-0.5451310134559277,-0.9120912508546379>, 0.5}
    cylinder { m*<-4.473284240111975,3.4785127368982143,-2.502119472324625>, <-1.370464507506571,-0.5451310134559277,-0.9120912508546379>, 0.5 }
    cylinder {  m*<-2.7005423621448097,-3.152818103165367,-1.567208968729166>, <-1.370464507506571,-0.5451310134559277,-0.9120912508546379>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    