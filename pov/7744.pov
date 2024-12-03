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
    sphere { m*<-0.4487225774521093,-0.49777038941586094,-0.4573618446178871>, 1 }        
    sphere {  m*<0.9704449167480523,0.49216852446405635,9.39192825241726>, 1 }
    sphere {  m*<8.338232115070848,0.20707627367179415,-5.178749176656671>, 1 }
    sphere {  m*<-6.557731078618144,6.730157647292434,-3.6879422734750644>, 1}
    sphere { m*<-3.8145014701372704,-7.827779623050604,-2.0160121437562344>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9704449167480523,0.49216852446405635,9.39192825241726>, <-0.4487225774521093,-0.49777038941586094,-0.4573618446178871>, 0.5 }
    cylinder { m*<8.338232115070848,0.20707627367179415,-5.178749176656671>, <-0.4487225774521093,-0.49777038941586094,-0.4573618446178871>, 0.5}
    cylinder { m*<-6.557731078618144,6.730157647292434,-3.6879422734750644>, <-0.4487225774521093,-0.49777038941586094,-0.4573618446178871>, 0.5 }
    cylinder {  m*<-3.8145014701372704,-7.827779623050604,-2.0160121437562344>, <-0.4487225774521093,-0.49777038941586094,-0.4573618446178871>, 0.5}

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
    sphere { m*<-0.4487225774521093,-0.49777038941586094,-0.4573618446178871>, 1 }        
    sphere {  m*<0.9704449167480523,0.49216852446405635,9.39192825241726>, 1 }
    sphere {  m*<8.338232115070848,0.20707627367179415,-5.178749176656671>, 1 }
    sphere {  m*<-6.557731078618144,6.730157647292434,-3.6879422734750644>, 1}
    sphere { m*<-3.8145014701372704,-7.827779623050604,-2.0160121437562344>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9704449167480523,0.49216852446405635,9.39192825241726>, <-0.4487225774521093,-0.49777038941586094,-0.4573618446178871>, 0.5 }
    cylinder { m*<8.338232115070848,0.20707627367179415,-5.178749176656671>, <-0.4487225774521093,-0.49777038941586094,-0.4573618446178871>, 0.5}
    cylinder { m*<-6.557731078618144,6.730157647292434,-3.6879422734750644>, <-0.4487225774521093,-0.49777038941586094,-0.4573618446178871>, 0.5 }
    cylinder {  m*<-3.8145014701372704,-7.827779623050604,-2.0160121437562344>, <-0.4487225774521093,-0.49777038941586094,-0.4573618446178871>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    