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
    sphere { m*<0.5550364859538506,1.1200090566082506,0.19404501200252233>, 1 }        
    sphere {  m*<0.7963478746130961,1.2286654062277491,3.1823451275015>, 1 }
    sphere {  m*<3.2895950636756304,1.2286654062277487,-1.0349370809891154>, 1 }
    sphere {  m*<-1.2998091049142522,3.793822330359051,-0.9026485767678118>, 1}
    sphere { m*<-3.9636363678015365,-7.391070539483407,-2.4770229703978055>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7963478746130961,1.2286654062277491,3.1823451275015>, <0.5550364859538506,1.1200090566082506,0.19404501200252233>, 0.5 }
    cylinder { m*<3.2895950636756304,1.2286654062277487,-1.0349370809891154>, <0.5550364859538506,1.1200090566082506,0.19404501200252233>, 0.5}
    cylinder { m*<-1.2998091049142522,3.793822330359051,-0.9026485767678118>, <0.5550364859538506,1.1200090566082506,0.19404501200252233>, 0.5 }
    cylinder {  m*<-3.9636363678015365,-7.391070539483407,-2.4770229703978055>, <0.5550364859538506,1.1200090566082506,0.19404501200252233>, 0.5}

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
    sphere { m*<0.5550364859538506,1.1200090566082506,0.19404501200252233>, 1 }        
    sphere {  m*<0.7963478746130961,1.2286654062277491,3.1823451275015>, 1 }
    sphere {  m*<3.2895950636756304,1.2286654062277487,-1.0349370809891154>, 1 }
    sphere {  m*<-1.2998091049142522,3.793822330359051,-0.9026485767678118>, 1}
    sphere { m*<-3.9636363678015365,-7.391070539483407,-2.4770229703978055>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7963478746130961,1.2286654062277491,3.1823451275015>, <0.5550364859538506,1.1200090566082506,0.19404501200252233>, 0.5 }
    cylinder { m*<3.2895950636756304,1.2286654062277487,-1.0349370809891154>, <0.5550364859538506,1.1200090566082506,0.19404501200252233>, 0.5}
    cylinder { m*<-1.2998091049142522,3.793822330359051,-0.9026485767678118>, <0.5550364859538506,1.1200090566082506,0.19404501200252233>, 0.5 }
    cylinder {  m*<-3.9636363678015365,-7.391070539483407,-2.4770229703978055>, <0.5550364859538506,1.1200090566082506,0.19404501200252233>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    