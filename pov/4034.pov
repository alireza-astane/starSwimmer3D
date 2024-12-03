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
    sphere { m*<-0.15284359682319998,-0.07693629761382012,-0.2525103145651393>, 1 }        
    sphere {  m*<0.10755154147239632,0.062285103617898296,2.9790280961969224>, 1 }
    sphere {  m*<2.581864797183057,0.025097677772553992,-1.481719840016323>, 1 }
    sphere {  m*<-1.7744589567160902,2.2515376468047785,-1.2264560799811095>, 1}
    sphere { m*<-1.5066717356782584,-2.636154295599119,-1.0369097948185368>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.10755154147239632,0.062285103617898296,2.9790280961969224>, <-0.15284359682319998,-0.07693629761382012,-0.2525103145651393>, 0.5 }
    cylinder { m*<2.581864797183057,0.025097677772553992,-1.481719840016323>, <-0.15284359682319998,-0.07693629761382012,-0.2525103145651393>, 0.5}
    cylinder { m*<-1.7744589567160902,2.2515376468047785,-1.2264560799811095>, <-0.15284359682319998,-0.07693629761382012,-0.2525103145651393>, 0.5 }
    cylinder {  m*<-1.5066717356782584,-2.636154295599119,-1.0369097948185368>, <-0.15284359682319998,-0.07693629761382012,-0.2525103145651393>, 0.5}

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
    sphere { m*<-0.15284359682319998,-0.07693629761382012,-0.2525103145651393>, 1 }        
    sphere {  m*<0.10755154147239632,0.062285103617898296,2.9790280961969224>, 1 }
    sphere {  m*<2.581864797183057,0.025097677772553992,-1.481719840016323>, 1 }
    sphere {  m*<-1.7744589567160902,2.2515376468047785,-1.2264560799811095>, 1}
    sphere { m*<-1.5066717356782584,-2.636154295599119,-1.0369097948185368>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.10755154147239632,0.062285103617898296,2.9790280961969224>, <-0.15284359682319998,-0.07693629761382012,-0.2525103145651393>, 0.5 }
    cylinder { m*<2.581864797183057,0.025097677772553992,-1.481719840016323>, <-0.15284359682319998,-0.07693629761382012,-0.2525103145651393>, 0.5}
    cylinder { m*<-1.7744589567160902,2.2515376468047785,-1.2264560799811095>, <-0.15284359682319998,-0.07693629761382012,-0.2525103145651393>, 0.5 }
    cylinder {  m*<-1.5066717356782584,-2.636154295599119,-1.0369097948185368>, <-0.15284359682319998,-0.07693629761382012,-0.2525103145651393>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    