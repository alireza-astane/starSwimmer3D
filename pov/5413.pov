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
    sphere { m*<-0.7816804909044088,-0.1577991549779471,-1.4540395937219435>, 1 }        
    sphere {  m*<0.3107048509912354,0.2859981677422461,8.476186211168072>, 1 }
    sphere {  m*<4.5225105807869985,0.032834098980217136,-4.026169841464727>, 1 }
    sphere {  m*<-2.428382092389504,2.170998835413808,-2.3841293273946538>, 1}
    sphere { m*<-2.1605948713516723,-2.7166931069900895,-2.1945830422320833>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3107048509912354,0.2859981677422461,8.476186211168072>, <-0.7816804909044088,-0.1577991549779471,-1.4540395937219435>, 0.5 }
    cylinder { m*<4.5225105807869985,0.032834098980217136,-4.026169841464727>, <-0.7816804909044088,-0.1577991549779471,-1.4540395937219435>, 0.5}
    cylinder { m*<-2.428382092389504,2.170998835413808,-2.3841293273946538>, <-0.7816804909044088,-0.1577991549779471,-1.4540395937219435>, 0.5 }
    cylinder {  m*<-2.1605948713516723,-2.7166931069900895,-2.1945830422320833>, <-0.7816804909044088,-0.1577991549779471,-1.4540395937219435>, 0.5}

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
    sphere { m*<-0.7816804909044088,-0.1577991549779471,-1.4540395937219435>, 1 }        
    sphere {  m*<0.3107048509912354,0.2859981677422461,8.476186211168072>, 1 }
    sphere {  m*<4.5225105807869985,0.032834098980217136,-4.026169841464727>, 1 }
    sphere {  m*<-2.428382092389504,2.170998835413808,-2.3841293273946538>, 1}
    sphere { m*<-2.1605948713516723,-2.7166931069900895,-2.1945830422320833>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3107048509912354,0.2859981677422461,8.476186211168072>, <-0.7816804909044088,-0.1577991549779471,-1.4540395937219435>, 0.5 }
    cylinder { m*<4.5225105807869985,0.032834098980217136,-4.026169841464727>, <-0.7816804909044088,-0.1577991549779471,-1.4540395937219435>, 0.5}
    cylinder { m*<-2.428382092389504,2.170998835413808,-2.3841293273946538>, <-0.7816804909044088,-0.1577991549779471,-1.4540395937219435>, 0.5 }
    cylinder {  m*<-2.1605948713516723,-2.7166931069900895,-2.1945830422320833>, <-0.7816804909044088,-0.1577991549779471,-1.4540395937219435>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    